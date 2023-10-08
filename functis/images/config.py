from enum import Enum
from typing import Any, Callable, Optional, Tuple
from warnings import warn

import cv2
import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel


def torch_permuter(img: torch.Tensor, channels: int = 3) -> torch.Tensor:
    if channels == 1 and 1 not in img.shape:
        return img
    elif channels == 1 and 1 in img.shape:
        return img.squeeze(img.shape.index(1))
    elif 1 in img.shape and channels != 1:
        idx_1 = img.shape.index(1)
        if idx_1 == 0:
            img = img.repeat(3, 1, 1)
        elif idx_1 == 1:
            warn(f"idx_1 == 1, this is probably wrong! {img.shape}")
            img = img.repeat(1, 3, 1)
        elif idx_1 == 2:
            img = img.repeat(1, 1, 3)
        else:
            raise ValueError(f"Invalid number of channels: {channels}, {img.shape}")
    if channels not in img.shape:
        if channels == 4 and 3 in img.shape:
            channel_index = img.shape.index(3)
            if channel_index == 0:
                img = torch.cat([img, torch.ones_like(img[:1, :, :]) * 255], dim=0)
            elif channel_index == 1:
                img = torch.cat([img, torch.ones_like(img[:, :1, :]) * 255], dim=1)
            elif channel_index == 2:
                img = torch.cat([img, torch.ones_like(img[:, :, :1]) * 255], dim=2)
            else:
                raise ValueError(f"Invalid number of channels: {channels}, {img.shape}")
    assert channels in img.shape, f"Invalid number of channels: {channels}, {img.shape}"
    return img, img.shape.index(channels)


class ImageLayout(str, Enum):
    hwc: str = "hwc"
    chw: str = "chw"

    def to_torch(self) -> Callable[[torch.Tensor], torch.Tensor]:
        if self == ImageLayout.hwc:

            def torch_permute(img: torch.Tensor, channels: int = 3) -> torch.Tensor:
                img = torch_permuter(img, channels)
                if channels == 1:
                    return img
                img, ch_idx = img
                channels_index = img.shape.index(channels)
                if channels_index == 0:
                    return img.permute(1, 2, 0)
                elif channels_index == 1:
                    warn(f"channels_index == 1, this is probably wrong! {img.shape}")
                    return img.permute(0, 2, 1)
                elif channels_index == 2:
                    return img

            return torch_permute
        elif self == ImageLayout.chw:

            def torch_permute(img: torch.Tensor, channels: int = 3) -> torch.Tensor:
                img = torch_permuter(img, channels)
                if channels == 1:
                    return img
                img, ch_idx = img
                channels_index = img.shape.index(channels)
                if channels_index == 0:
                    return img
                elif channels_index == 1:
                    warn("channels_index == 1, this is probably wrong!")
                    return img.permute(1, 0, 2)
                elif channels_index == 2:
                    return img.permute(2, 0, 1)

            return torch_permute
        else:
            raise ValueError(f"Invalid image layout: {self}")

    def to_np(self) -> Callable[[np.ndarray], np.ndarray]:
        def np_transpose(
            img: np.ndarray, channels: int = 3, color_hint="rgb"
        ) -> np.ndarray:
            if channels == 1 and 1 not in img.shape:
                return img
            elif channels == 1 and 1 in img.shape:
                return img.squeeze(img.shape.index(1))
            elif len(img.shape) == 2 and channels == 3:
                img = (
                    cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    if color_hint == "rgb"
                    else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                )
            elif len(img.shape) == 2 and channels == 4:
                img = (
                    cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
                    if color_hint == "rgb"
                    else cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                )
            elif channels == 3 and img.shape[-1] == 3:
                img = (
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if color_hint == "rgb" else img
                )
            elif channels == 4 and img.shape[-1] == 3:
                img = (
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
                    if color_hint == "rgb"
                    else cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                )
            elif channels == 3 and img.shape[-1] == 4:
                img = (
                    cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                    if color_hint == "rgb"
                    else cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                )
            elif channels == 4 and img.shape[-1] == 4:
                img = (
                    cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                    if color_hint == "rgb"
                    else img
                )
            else:
                raise ValueError(f"Invalid number of channels: {channels}, {img.shape}")

            assert (
                channels in img.shape
            ), f"Unexpected number of channels in image, expected {channels}, got: {img.shape}"
            if self == ImageLayout.chw:
                img = img.transpose(2, 0, 1)
            return img

        return np_transpose

    def to_pil(self) -> Callable[[Image.Image], Image.Image]:
        def permute_fn(x):
            return x

        return permute_fn


class ReadMethod(str, Enum):
    pil: str = "pil"
    torch: str = "torch"
    np: str = "np"  # opencv
    nvjpeg: str = "nvjpeg"


class DataType(str, Enum):
    uint8 = "uint8"
    uint16 = "uint16"
    float32 = "float32"
    float16 = "float16"
    pil_image = "pil_image"

    def to_torch(self) -> torch.dtype:
        if self == DataType.uint8:
            return torch.uint8
        elif self == DataType.uint16:
            return torch.uint16
        elif self == DataType.float32:
            return torch.float32
        elif self == DataType.float16:
            return torch.float16
        else:
            raise ValueError(f"Invalid data type: {self}")

    def to_np(self) -> np.dtype:
        if self == DataType.uint8:
            return np.uint8
        elif self == DataType.uint16:
            return np.uint16
        elif self == DataType.float32:
            return np.float32
        elif self == DataType.float16:
            return np.float16
        else:
            raise ValueError(f"Invalid data type: {self}")

    def to_pil(self) -> str:
        if self == DataType.uint8:
            return "L"
        elif self == DataType.uint16:
            return "I;16"
        elif self == DataType.float32:
            return "F"
        elif self == DataType.float16:
            return "F;16"
        elif self == DataType.pil_image:
            return ""
        else:
            raise ValueError(f"Invalid data type: {self}")


class ImageType(Enum):
    np = np.ndarray
    torch = torch.Tensor
    pil = Image.Image


class ColorMode(Enum):
    rgb: str = "RGB"
    rgba: str = "RGBA"
    bgr: str = "BGR"
    bgra: str = "BGRA"
    grayscale: str = "L"

    def to_np(self) -> Tuple[int, int, int]:
        """
        Returns a tuple of (cv2.imread flag, number of channels, cv2.COLOR_*2*RGB)
        """
        if self == ColorMode.rgb:
            return (cv2.IMREAD_COLOR, 3, cv2.COLOR_BGR2RGB)
        elif self == ColorMode.rgba:
            return (cv2.IMREAD_COLOR, 4, cv2.COLOR_BGRA2RGBA)
        elif self == ColorMode.bgr:
            return (cv2.IMREAD_COLOR, 3, None)
        elif self == ColorMode.bgra:
            return (cv2.IMREAD_COLOR, 4, None)
        elif self == ColorMode.grayscale:
            return (cv2.IMREAD_GRAYSCALE, 1, None)
        else:
            raise ValueError(f"Invalid color mode: {self}")

    def to_torch(self):
        return self

    def to_pil(self):
        if self == ColorMode.rgb:
            return "RGB"
        elif self == ColorMode.rgba:
            return "RGBA"
        elif self == ColorMode.bgr:
            return "BGR"
        elif self == ColorMode.bgra:
            return "BGRA"
        elif self == ColorMode.grayscale:
            return "L"
        else:
            raise ValueError(f"Invalid color mode: {self}")


class Device(Enum):
    cpu: str = "cpu"
    cuda: str = "cuda"

    def to_torch(self, idx: Optional[int] = None):
        if self == Device.cpu:
            return torch.device(
                "cpu",
            )
        elif self == Device.cuda:
            return torch.device("cuda", idx)
        else:
            raise ValueError(f"Invalid device: {self}")

    def to_np(self):
        if self == Device.cpu:
            return
        elif self == Device.cuda:
            raise NotImplementedError("cuda not implemented for numpy")
        else:
            raise ValueError(f"Invalid device: {self}")


class SpecificConfig:
    data_type: Any
    device: Any
    color_mode: Any
    read_method: Any
    image_layout: Any

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class ReadConfig(BaseModel):
    data_type: DataType = DataType.uint8
    device: Optional[Device] = None
    color_mode: ColorMode = ColorMode.rgb
    read_method: ReadMethod = ReadMethod.torch
    image_layout: ImageLayout = ImageLayout.hwc
    image_extensions: Tuple[str, ...] = (
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "tif",
        "tiff",
    )

    def to_torch(self, idx: Optional[int] = None) -> "SpecificConfig":
        return SpecificConfig(
            data_type=self.data_type.to_torch(),
            device=self.device.to_torch(idx) if self.device else torch.device("cpu"),
            color_mode=self.color_mode.to_torch(),
            image_layout=self.image_layout.to_torch(),
            read_method=self.read_method
            if self.read_method in [ReadMethod.torch, ReadMethod.nvjpeg]
            else ReadMethod.torch,
        )

    def to_np(self) -> "SpecificConfig":
        return SpecificConfig(
            data_type=self.data_type.to_np(),
            device=None,
            color_mode=self.color_mode.to_np(),
            image_layout=self.image_layout.to_np(),
            read_method=ReadMethod.np,
        )

    def to_pil(self) -> "SpecificConfig":
        return SpecificConfig(
            data_type=self.data_type.to_pil(),
            device=None,
            color_mode=self.color_mode.to_pil(),
            image_layout=self.image_layout.to_pil(),
            read_method=ReadMethod.pil,
        )

    def to_specific_config(self) -> "SpecificConfig":
        if self.read_method == ReadMethod.torch:
            return self.to_torch()
        elif self.read_method == ReadMethod.np:
            return self.to_np()
        elif self.read_method == ReadMethod.pil:
            return self.to_pil()
        elif self.read_method == ReadMethod.nvjpeg:
            return self.to_torch()
        else:
            raise ValueError(f"Invalid read method: {self.read_method}")

    model_config = {
        "arbitrary_types_allowed": True,
    }
