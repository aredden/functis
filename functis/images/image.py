import io
from pathlib import Path
from typing import Generator, List, Optional, Tuple, TypeVar, Union, overload

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import io as tio
from yaml import load as yaml_load

from .warns import warn_once

has_nvjpeg = False
try:
    from nvjpeg_torch import Jpeg as NVJpeg

    has_nvjpeg = True
except ImportError:
    NVJpeg = None

import imghdr

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
    _nvjpeg: NVJpeg
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

    def with_format(self, image_layout: Union[ImageLayout, str]) -> "ImageIO":
        return self.with_layout(image_layout)

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

    def read(self, path: Union[bytes, str, Path], **kwargs):
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
        try:
            image = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR) if not isinstance(path,bytes) else cv2.imdecode(np.frombuffer(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cv2.imread returned None for file: {path}")
            if len(image.shape) > 2:
                if image.shape[-1] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                elif image.shape[-1] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    raise ValueError(
                        f"Unsupported number of color channels! {image.shape[-1]}, only 3 or 4 (or grayscale) are supported."
                    )
            else:
                image = np.ascontiguousarray(image).copy()
        except Exception as e:
            warn_once(
                f"Error reading image with cv2: {e}, falling back to PIL for file: {path}"
            )
            if isinstance(path,bytes):
                image = Image.open(io.BytesIO(path))
            else:
                image = Image.open(path.as_posix())
            if len(image.split()) == 2:
                image = image.convert("RGB")
            elif len(image.split()) == 4:
                image = image.convert("RGBA")
            elif len(image.split()) == 3:
                image = image.convert("RGB")
            else:
                raise ValueError(
                    f"Unsupported number of color channels! {len(image.split())}, only 3 or 4 (or grayscale) are supported."
                )
            image = np.asarray(image).copy()
        if len(image.shape) == 2:
            if self.config.color_mode.name == ColorMode.grayscale.name:
                return image
            image = image[..., None]
        elif image.shape[-1] == 4 and converter.channels == 3:
            image = image[..., :3]
        elif image.shape[-1] == 3 and converter.channels == 4:
            if self.config.color_mode.name.lower().startswith("bgr"):
                image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA, image)
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
        type_of_file = None

        is_bytes = isinstance(path, bytes)
        printable = 'bytes' if is_bytes else path
        if is_bytes:
            type_of_file = imghdr.what(None, path) 
            if type_of_file is None:
                raise ValueError(f"Invalid image type for bytes!")
            if type_of_file[0] != ".":
                type_of_file = "." + type_of_file
        
        suffix = path.suffix.lower() if not is_bytes else type_of_file

        path = Path(path) if not isinstance(path, Path) and not is_bytes else path

        if using_nvjpeg and not self._nvjpeg_available:
            warn_once(
                f"nvjpeg not available, falling back to torchvision.io for image reading."
            )
            self.config.read_method = ReadMethod.torch
            return self._read_torch(
                path, specific_config, converter, using_nvjpeg=False
            )

        if converter is None:
            converter = parse_colormode_torch(specific_config.color_mode)

        if using_nvjpeg and suffix not in [".jpg", ".jpeg"]:
            warn_once(
                f"nvjpeg only supports jpeg images, falling back to torchvision for file: {printable}"
            )
            return self._read_torch(
                path, specific_config, converter, using_nvjpeg=False
            )
        if isinstance(path,bytes):
            if suffix in [".jpg", ".jpeg"] and not using_nvjpeg and is_bytes:
                image_bytes_torch = torch.from_numpy(np.frombuffer(path, dtype=np.uint8).copy())
            elif not using_nvjpeg:
                image_bytes_torch = torch.from_numpy(np.frombuffer(path,dtype=np.uint8).copy())
        else:
            image_bytes_torch = tio.read_file(path.as_posix()) if not using_nvjpeg else None


        if suffix in [".jpg", ".jpeg"] and not using_nvjpeg:
            try:
                image = tio.decode_jpeg(
                    image_bytes_torch,
                    mode=converter.mode,
                    device=specific_config.device,
                )
            except RuntimeError as e:
                if "The provided mode is not supported" in str(e):
                    warn_once(
                        f"Mode {converter.mode} not supported by torchvision.io.decode_jpeg, falling back to cv2.imread for file: {printable}"
                    )
                    image = self._read_fallback_torch(path, converter, specific_config)
                else:
                    raise e
        elif suffix in [".jpg", ".jpeg"] and using_nvjpeg:
            try:
                image = self._nvjpeg.read_image(
                    path.as_posix(), self._device_for_nvjpeg
                ) if not is_bytes else self._nvjpeg.decode(path)
            except RuntimeError as e:
                warn_once(
                    f"Error reading image from nvjpeg torch: {e}, falling back to reading with torchvision for file: {printable}"
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
        is_bytes = isinstance(path, bytes)
        path = Path(path) if not isinstance(path, Path) and not is_bytes else path
        hint = "bgr" if self.config.color_mode.name.lower().startswith("bgr") else "rgb"
        readmode, channels, flip = specific_config.color_mode
        image = cv2.imread(path.as_posix(), readmode) if not isinstance(path,bytes) else cv2.imdecode(np.frombuffer(path, dtype=np.uint8), readmode)
        image = specific_config.image_layout(image, channels, hint)
        return image.astype(specific_config.data_type)

    def _read_pil(
        self, path: Union[str, Path], specific_config: SpecificConfig
    ) -> Image.Image:
        is_bytes = isinstance(path, bytes)
        path = Path(path) if not isinstance(path, Path) and not is_bytes else path

        im = Image.open(path.as_posix()) if not is_bytes else Image.open(io.BytesIO(path))
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
        extra_kwargs = {
            "specific_config": self.config.to_specific_config(),
        }
        if self.config.read_method in [ReadMethod.torch,ReadMethod.nvjpeg]:
            extra_kwargs["converter"] = parse_colormode_torch(
                self.config.color_mode
            )
        if recursive:
            for path in dirpath.rglob("*"):
                if (
                    path.is_file()
                    and path.suffix.lower().lstrip(".") in self.config.image_extensions
                ):
                    if return_paths:
                        yield path, self.read(path, **extra_kwargs)
                    else:
                        yield self.read(path, **extra_kwargs)
        else:
            for path in dirpath.glob("*"):
                if (
                    path.is_file()
                    and path.suffix.lower().lstrip(".") in self.config.image_extensions
                ):
                    if return_paths:
                        yield path, self.read(path, **extra_kwargs)
                    else:
                        yield self.read(path, **extra_kwargs)

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
        extra_kwargs = {}
        if self.config.read_method in [ReadMethod.torch,ReadMethod.nvjpeg]:
            extra_kwargs["converter"] = parse_colormode_torch(
                self.config.color_mode
            )
        for path in paths:
            path = Path(path) if not isinstance(path, Path) else path
            if (
                path.is_file()
                and path.suffix.lower().lstrip(".") in self.config.image_extensions
            ):
                if return_paths:
                    yield path, self.read(path, **extra_kwargs)
                else:
                    yield self.read(path, **extra_kwargs)
            else:
                warn_once(f"Invalid image path, will ignore: {path}", UserWarning)
