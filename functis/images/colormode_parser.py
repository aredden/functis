from typing import Literal
import torch
from torchvision import io as tio

from .config import ColorMode


class Converter:
    def __init__(self, mode, flip, channels) -> None:
        self.mode = mode
        self.flip = flip
        self.channels = channels

    def to_grayscale(self, img: torch.Tensor, layout: Literal["hwc", "chw"]):
        img, _ = (
            (img[:3].unbind(0), img[3:])
            if layout == "chw"
            else (img[..., :3].unbind(-1), img[..., 3:])
        )
        R, G, B = img
        L = R.float() * 299 / 1000 + G.float() * 587 / 1000 + B.float() * 114 / 1000
        L = L.clamp(0, 255).type(torch.uint8)
        return L

    def to_readmode(self):
        return self.mode

    def do_conversion(self, img: torch.Tensor):
        channels = img.shape[0] if len(img.shape) == 3 else 1
        if channels != self.channels:
            # convert to correct number of channels
            if channels == 1 and self.channels != 1:
                img = img.repeat(3, 1, 1)
                if self.channels == 4:
                    img = torch.cat([img, torch.ones_like(img[:1, ...])], dim=0)
                return img

            elif channels in [3, 4] and self.channels == 1:
                img = self.to_grayscale(img, "chw")
            elif channels == 4 and self.channels == 3:
                img = img[:3, ...]
            elif channels == 3 and self.channels == 4:
                img = torch.cat([img, torch.ones_like(img[:1, ...]) * 255], dim=0)
            else:
                raise ValueError(f"Invalid number of channels: {channels}, {img.shape}")
        if self.channels == 1:
            if 1 in img.shape:
                return img.squeeze(img.shape.index(1))
            else:
                return img
        elif self.channels == 3:
            if self.flip:
                return img.flip(0)
            else:
                return img
        elif self.channels == 4:
            if self.flip:
                return torch.cat([img[:3, :, :].flip(0), img[3:, :, :]], dim=0)
            else:
                return img
        else:
            raise ValueError(f"Invalid number of channels: {channels}")

    def do_conversion_nvjpeg(self, img: torch.Tensor):
        """
        NVJpeg reads images as RGB in HWC format with uint8 dtype, requires special handling
        """
        channels = img.shape[-1] if len(img.shape) == 3 else 1
        if channels != self.channels:
            # convert to correct number of channels
            if channels == 1 and self.channels != 1:
                img = img.repeat(1, 1, 3)
                if self.channels == 4:
                    img = torch.cat([img, torch.ones_like(img[..., :1])], dim=-1)
                return img

            elif channels in [3, 4] and self.channels == 1:
                img = self.to_grayscale(img, "hwc")
            elif channels == 4 and self.channels == 3:
                img = img[..., :3]
            elif channels == 3 and self.channels == 4:
                img = torch.cat([img, torch.ones_like(img[..., :1]) * 255], dim=-1)
            else:
                raise ValueError(f"Invalid number of channels: {channels}, {img.shape}")
        if self.channels == 1:
            if 1 in img.shape:
                return img.squeeze(img.shape.index(1))
            else:
                return img
        elif self.channels == 3:
            if self.flip:
                return img.flip(-1)
            else:
                return img
        elif self.channels == 4:
            if self.flip:
                return torch.cat([img[:, :, :3].flip(-1), img[:, :, 3:]], dim=-1)
            else:
                return img
        else:
            raise ValueError(f"Invalid number of channels: {channels}")


def parse_colormode_torch(colormode: ColorMode) -> Converter:
    """
    Returns class for converting color mode for torchvision.io.read_image and whether to convert to BGR
    """
    if colormode == ColorMode.bgr:
        mode, flip, channels = tio.image.ImageReadMode.RGB, True, 3
    elif colormode == ColorMode.rgb:
        mode, flip, channels = tio.image.ImageReadMode.RGB, False, 3
    elif colormode == ColorMode.rgba:
        mode, flip, channels = tio.image.ImageReadMode.RGB_ALPHA, False, 4
    elif colormode == ColorMode.bgra:
        mode, flip, channels = tio.image.ImageReadMode.RGB_ALPHA, True, 4
    elif colormode == ColorMode.grayscale:
        mode, flip, channels = tio.image.ImageReadMode.GRAY, False, 1
    else:
        raise ValueError(f"Invalid color mode: {colormode}")

    return Converter(mode, flip, channels)
