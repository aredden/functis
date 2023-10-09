from pathlib import Path
from typing import (
    Generator,
    Generic,
    List,
    Optional,
    Tuple,
    Union,
    TypeVar,
    overload,
    TypeGuard,
)
from warnings import warn

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import io as tio
from yaml import load as yaml_load

has_nvjpeg = False
try:
    from nvjpeg_torch import Jpeg as NVJpeg

    has_nvjpeg = True
except ImportError:
    NVJpeg = None


from .colormode_parser import Converter, parse_colormode_torch
from .config import (
    ColorMode,
    DataType,
    Device,
    ImageLayout,
    ReadConfig,
    ReadMethod,
    SpecificConfig,
)

UsesPath = TypeVar("UsesPath")

ImageT = TypeVar("ImageT", Image.Image, np.ndarray, torch.Tensor)


class ImageIOBase:
    def __init__(self, read_config: ReadConfig, **kwargs) -> None:
        self.config = read_config
        self._device_idx = kwargs.get("idx", 0)

    @classmethod
    def from_kwargs(cls, **kwargs) -> "ImageIO":
        return cls(ReadConfig(**kwargs))

    @classmethod
    def from_config_json(cls, config_file: Union[str, Path]) -> "ImageIO":
        return cls(ReadConfig.model_validate_json(open(config_file, "rw").read()))

    @classmethod
    def from_config_yaml(cls, config_file: Union[str, Path]) -> "ImageIO":
        json_dict = yaml_load(open(config_file, "rw").read(), Loader=yaml.FullLoader)
        return cls(ReadConfig.model_validate(json_dict))


class ImageIO(ImageIOBase):
    def __init__(self, read_config: ReadConfig = None, **kwargs) -> None:
        if read_config is None:
            read_config = ReadConfig()
        self.config = read_config
        self._device_idx = kwargs.get("idx", 0)
        self._init_nvjpeg(self.config.read_method)

    def _init_nvjpeg(self, read_method: ReadMethod):
        self._device_for_nvjpeg = (
            torch.device("cuda", self._device_idx)
            if torch.cuda.is_available()
            else None
        )
        self._nvjpeg_available = has_nvjpeg and self._device_for_nvjpeg is not None
        if read_method == ReadMethod.nvjpeg and self._nvjpeg_available:
            self._nvjpeg: NVJpeg = NVJpeg(self._device_for_nvjpeg)
        else:
            self._nvjpeg: NVJpeg = None

    def with_config(self, config: ReadConfig) -> "ImageIO":
        self.config = config
        self._init_nvjpeg(self.config.read_method)
        return self

    def with_device(
        self, device: Union[Device, str, torch.device], idx: Optional[int] = 0
    ) -> "ImageIO":
        self._device_idx = idx
        if isinstance(device, str):
            if ":" in device:
                device, idx = device.split(":")
                self._device_idx = int(idx)
            self.config.device = Device(device)
        elif isinstance(device, Device):
            self.config.device = device
        elif isinstance(device, torch.device):
            self.config.device = Device(device.type)
            self._device_idx = device.index
        else:
            raise ValueError(f"Invalid device: {device}")
        self._init_nvjpeg(self.config.read_method)
        return self

    def with_color(self, color_mode: Union[ColorMode, str]) -> "ImageIO":
        if isinstance(color_mode, str):
            self.config.color_mode = ColorMode(color_mode.upper())
        elif isinstance(color_mode, ColorMode):
            self.config.color_mode = color_mode
        else:
            raise ValueError(f"Invalid color_mode: {color_mode}")
        return self

    def with_layout(self, image_layout: Union[ImageLayout, str]) -> "ImageIO":
        if isinstance(image_layout, str):
            self.config.image_layout = ImageLayout(image_layout)
        elif isinstance(image_layout, ImageLayout):
            self.config.image_layout = image_layout
        else:
            raise ValueError(f"Invalid image_layout: {image_layout}")
        return self

    def with_reader(self, read_method: Union[ReadMethod, str]) -> "ImageIO":
        if isinstance(read_method, str):
            self.config.read_method = ReadMethod(read_method)
        elif isinstance(read_method, ReadMethod):
            self.config.read_method = read_method
        else:
            raise ValueError(f"Invalid read_method: {read_method}")
        self._init_nvjpeg(self.config.read_method)
        return self

    def with_dtype(self, data_type: Union[DataType, str]) -> "ImageIO":
        if isinstance(data_type, str):
            self.config.data_type = DataType(data_type)
        elif isinstance(data_type, DataType):
            self.config.data_type = data_type
        else:
            raise ValueError(f"Invalid data_type: {data_type}")
        return self

    def read(self, path: Union[str, Path], **kwargs):
        specific_config = (
            kwargs.get("specific_config", None) or self.config.to_specific_config()
        )
        if self.config.read_method == ReadMethod.torch:
            converter = kwargs.get("converter", None) or parse_colormode_torch(
                specific_config.color_mode
            )
            return self._read_torch(
                path, specific_config, converter, using_nvjpeg=False
            )
        elif self.config.read_method == ReadMethod.np:
            return self._read_np(path, specific_config)
        elif self.config.read_method == ReadMethod.pil:
            return self._read_pil(path, specific_config)
        elif self.config.read_method == ReadMethod.nvjpeg:
            converter = kwargs.get("converter", None) or parse_colormode_torch(
                specific_config.color_mode
            )
            return self._read_torch(path, specific_config, converter, using_nvjpeg=True)
        else:
            raise ValueError(f"Invalid read_method: {self.config.read_method}")

    def _read_fallback_torch(
        self, path: Path, converter: Converter, specific_config: SpecificConfig
    ):
        image = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
        if len(image.shape) == 2:
            if self.config.color_mode.name == ColorMode.grayscale.name:
                return image
            image = image[..., None]
        elif image.shape[-1] == 4 and converter.channels == 3:
            image = image[..., :3]
        elif image.shape[-1] == 3 and converter.channels == 4:
            if self.config.color_mode.name.lower().startswith("bgr"):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA, image)
            elif self.config.color_mode.name.lower().startswith("rgb"):
                image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA, image)
            else:
                raise ValueError(f"Invalid color mode: {self.config.color_mode}")
        image = torch.from_numpy(image).to(specific_config.device).permute(2, 0, 1)
        return image

    def _read_torch(
        self,
        path: Union[str, Path],
        specific_config: SpecificConfig,
        converter: Converter = None,
        using_nvjpeg: bool = False,
    ) -> torch.Tensor:
        path = Path(path) if not isinstance(path, Path) else path

        if converter is None:
            converter = parse_colormode_torch(specific_config.color_mode)

        if using_nvjpeg and path.suffix.lower() not in [".jpg", ".jpeg"]:
            warn(
                f"nvjpeg only supports jpeg images, falling back to torchvision for file: {path}"
            )
            return self._read_torch(
                path, specific_config, converter, using_nvjpeg=False
            )
        image_bytes_torch = tio.read_file(path.as_posix()) if not using_nvjpeg else None

        suffix = path.suffix.lower()

        if suffix in [".jpg", ".jpeg"] and not using_nvjpeg:
            try:
                image = tio.decode_jpeg(
                    image_bytes_torch,
                    mode=converter.mode,
                    device=specific_config.device,
                )
            except RuntimeError as e:
                if "The provided mode is not supported" in str(e):
                    warn(
                        f"Mode {converter.mode} not supported by torchvision.io.decode_jpeg, falling back to cv2.imread for file: {path}"
                    )
                    image = self._read_fallback_torch(path, converter, specific_config)
                else:
                    raise e
        elif suffix in [".jpg", ".jpeg"] and using_nvjpeg:
            try:
                image = self._nvjpeg.read_image(
                    path.as_posix(), self._device_for_nvjpeg
                )
            except RuntimeError as e:
                warn(
                    f"Error reading image from nvjpeg torch: {e}, falling back to reading with torchvision for file: {path}"
                )
                # This is okay because if torchvision fails, it will fall back to cv2 to read the image.
                return self._read_torch(
                    path, specific_config, converter, using_nvjpeg=False
                )

        elif suffix == ".png":
            image = tio.decode_png(image_bytes_torch, mode=converter.mode).to(
                specific_config.device
            )
        else:
            try:
                image = tio.decode_image(image_bytes_torch, mode=converter.mode).to(
                    specific_config.device
                )
            except RuntimeError as e:
                image = self._read_fallback_torch(path, converter, specific_config)

        if using_nvjpeg:
            image = converter.do_conversion_nvjpeg(image)
        else:
            image = converter.do_conversion(image)

        image = specific_config.image_layout(image, converter.channels)

        dtype: torch.dtype = specific_config.data_type
        im_dtype: torch.dtype = image.dtype
        if dtype == im_dtype:
            return image
        elif dtype == torch.uint8 and im_dtype.is_floating_point:
            if image.max() <= 1.0:
                return (image * 255.0).clamp(0, 255).type(torch.uint8)
            else:
                return image.to(torch.uint8) if im_dtype != torch.uint8 else image
        elif dtype == torch.float32:
            return image.float() if im_dtype != torch.float32 else image
        elif dtype == torch.float16:
            return image.half() if im_dtype != torch.float16 else image
        else:
            raise ValueError(f"Invalid data_type: {dtype}")

    def _read_np(
        self, path: Union[str, Path], specific_config: SpecificConfig
    ) -> np.ndarray:  # -> Any:
        path = Path(path) if not isinstance(path, Path) else path
        hint = "bgr" if self.config.color_mode.name.lower().startswith("bgr") else "rgb"
        readmode, channels, flip = specific_config.color_mode
        image = cv2.imread(path.as_posix(), readmode)
        image = specific_config.image_layout(image, channels, hint)
        return image.astype(specific_config.data_type)

    def _read_pil(
        self, path: Union[str, Path], specific_config: SpecificConfig
    ) -> Image.Image:
        path = Path(path) if not isinstance(path, Path) else path
        im = Image.open(path.as_posix())
        if specific_config.color_mode.upper() == "BGRA":
            if len(im.split()) == 1:
                npim = np.array(im)
                npim = cv2.cvtColor(npim, cv2.COLOR_GRAY2BGRA, npim)
                im = Image.fromarray(npim)
            elif len(im.split()) == 3:
                npim = np.array(im)
                npim = cv2.cvtColor(npim, cv2.COLOR_RGB2BGRA, npim)
                im = Image.fromarray(npim)
            elif len(im.split()) == 4:
                npim = np.array(im)
                npim = cv2.cvtColor(npim, cv2.COLOR_RGBA2BGRA, npim)
                im = Image.fromarray(npim)
            else:
                raise ValueError(
                    f"Invalid number of image dimensions: {len(im.split())}"
                )
        elif specific_config.color_mode.upper() == "BGR":
            if len(im.split()) == 1:
                npim = np.array(im)
                npim = cv2.cvtColor(npim, cv2.COLOR_GRAY2BGR, npim)
                im = Image.fromarray(npim)
            elif len(im.split()) == 4:
                npim = np.array(im)
                npim = cv2.cvtColor(npim, cv2.COLOR_RGBA2BGR, npim)
                im = Image.fromarray(npim)
            elif len(im.split()) == 3:
                npim = np.array(im)
                npim = cv2.cvtColor(npim, cv2.COLOR_RGB2BGR, npim)
                im = Image.fromarray(npim)
            else:
                raise ValueError(
                    f"Invalid number of image dimensions: {len(im.split())}"
                )
        else:
            im = im.convert(specific_config.color_mode)

        return im

    @overload
    def read_dir(
        self, dirpath: Union[str, Path], recursive: bool, return_paths: bool = False
    ) -> Generator[Union[Image.Image, torch.Tensor, np.ndarray], None, None]:
        ...

    @overload
    def read_dir(
        self, dirpath: Union[str, Path], recursive: bool, return_paths: bool = True
    ) -> Generator[
        Tuple[Path, Union[Image.Image, torch.Tensor, np.ndarray]], None, None
    ]:
        ...

    def read_dir(
        self, dirpath: Union[str, Path], recursive: bool = False, return_paths=False
    ):
        dirpath = Path(dirpath) if not isinstance(dirpath, Path) else dirpath
        if recursive:
            for path in dirpath.rglob("*"):
                if (
                    path.is_file()
                    and path.suffix.lower().lstrip(".") in self.config.image_extensions
                ):
                    if return_paths:
                        yield path, self.read(path)
                    else:
                        yield self.read(path)
        else:
            for path in dirpath.glob("*"):
                if (
                    path.is_file()
                    and path.suffix.lower().lstrip(".") in self.config.image_extensions
                ):
                    if return_paths:
                        yield path, self.read(path)
                    else:
                        yield self.read(path)

    @overload
    def read_list(
        self, paths: List[Union[str, Path]], return_paths=False
    ) -> Generator[Union[Image.Image, torch.Tensor, np.ndarray], None, None]:
        ...

    @overload
    def read_list(
        self, paths: List[Union[str, Path]], return_paths=True
    ) -> Generator[
        Tuple[Path, Union[Image.Image, torch.Tensor, np.ndarray]], None, None
    ]:
        ...

    def read_list(self, paths: List[Union[str, Path]], return_paths=False):
        for path in paths:
            path = Path(path) if not isinstance(path, Path) else path
            if (
                path.is_file()
                and path.suffix.lower().lstrip(".") in self.config.image_extensions
            ):
                if return_paths:
                    yield path, self.read(path)
                else:
                    yield self.read(path)
            else:
                warn(f"Invalid image path, will ignore: {path}", UserWarning)
