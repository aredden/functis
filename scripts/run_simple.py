import sys
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.as_posix()))
from functis import ColorMode, ReadConfig, ReadMethod
import torch
parent_dir = Path(__file__).parent.parent
test_image_dir = parent_dir / "tests" / "test_images"

def generate_read_configs():
    for read_method in ["torch", "np", "pil", "nvjpeg"]:
        for color_mode in ["rgb", "bgr", "l", "rgba", "bgra"]:
            for image_layout in ["hwc", "chw"]:
                for data_type in ["uint8", "float32", "float16", "pil_image"]:
                    for device in ["cpu", "cuda"]:
                        if read_method == "nvjpeg" and device == "cpu":
                            continue
                        if device == "cuda" and read_method in ["pil", "np"]:
                            continue
                        if data_type == "pil_image" and read_method != "pil":
                            continue
                        yield ReadConfig(
                            read_method=read_method,
                            color_mode=color_mode.upper(),
                            image_layout=image_layout,
                            data_type=data_type,
                            device=device
                        )

dtype_to_np_dtype = {
    "uint8": np.uint8,
    "float32": np.float32,
    "float16": np.float16,
    "uint16": np.uint16,
    "uint32": np.uint32,
}
dtype_to_torch_dtype = {
    "uint8": torch.uint8,
    "float32": torch.float32,
    "float16": torch.float16,
    # "uint16": torch.uint16,
    # "uint32": torch.uint32,
}
def test_read_image(image_path, config: ReadConfig):
    from functis.images.image import ImageIO
    img = ImageIO(config).read(image_path)
    if config.color_mode == ColorMode.grayscale and config.read_method != ReadMethod.pil:
        assert len(img.shape) == 2, f"{img.shape} != 2 for {config} and file {image_path}"
    elif config.color_mode in [ColorMode.rgb, ColorMode.bgr] and config.read_method != ReadMethod.pil:
        assert len(img.shape) == 3, f"{img.shape} != 3 for {config} and file {image_path}"
        assert 3 in img.shape, f"{img.shape} != 3 for {config} and file {image_path}"
    elif config.color_mode in [ColorMode.rgba, ColorMode.bgra]  and config.read_method != ReadMethod.pil:
        assert len(img.shape) == 3, f"{img.shape} != 4 for {config} and file {image_path}"
        assert 4 in img.shape, f"{img.shape} != 4 for {config} and file {image_path}"
    elif not config.read_method == ReadMethod.pil:
        raise ValueError(f"Invalid color mode: {config.color_mode}")
    if config.image_layout == "hwc" and config.color_mode != ColorMode.grayscale and config.read_method != ReadMethod.pil:
        assert img.shape[0] == 256, f"{img.shape[0]} != 256 for {config} and file {image_path}"
        assert img.shape[1] == 256, f"{img.shape[1]} != 256 for {config} and file {image_path}"
    elif config.image_layout == "chw" and config.color_mode != ColorMode.grayscale and config.read_method != ReadMethod.pil:
        assert img.shape[1] == 256, f"{img.shape[1]} != 256 for {config} and file {image_path}"
        assert img.shape[2] == 256, f"{img.shape} != (256, 256) for {config} and file {image_path}"
    elif config.read_method != ReadMethod.pil:
        assert img.shape == (256, 256), f"{img.shape} != (256, 256)"
    if config.read_method == "torch":
        assert isinstance(img, torch.Tensor), f"{type(img)} is not torch.Tensor for {config} and file {image_path}"
        assert img.device.type == config.device.to_torch().type, f"{img.device} != {config.device} for {config} and file {image_path}"
        assert img.dtype == dtype_to_torch_dtype[config.data_type], f"{img.dtype} != {dtype_to_torch_dtype[config.data_type]} for {config} and file {image_path}"
    elif config.read_method == "np":
        assert isinstance(img, np.ndarray), f"{type(img)} is not np.ndarray for {config} and file {image_path}"
        assert img.dtype == dtype_to_np_dtype[config.data_type], f"{img.dtype} != {dtype_to_np_dtype[config.data_type]} for {config} and file {image_path}"
    elif config.read_method == "nvjpeg":
        assert isinstance(img, torch.Tensor)
        assert img.device.type == config.device.to_torch().type, f"{img.device} != {config.device} for {config} and file {image_path}"
        assert img.dtype == dtype_to_torch_dtype[config.data_type], f"{img.dtype} != {dtype_to_torch_dtype[config.data_type]} for {config} and file {image_path}"
    else:
        assert isinstance(img, Image.Image), f"{type(img)} is not Image.Image for {config} and file {image_path}"
        if config.color_mode.name.upper().startswith("BGR"):
            ...
        else:
            assert img.mode == config.color_mode.value, f"{img.mode} != {config.color_mode} for {config} and file {image_path}"

if __name__ == "__main__":
    total_configs_to_test = len(list(test_image_dir.glob("*.png"))) * len(list(generate_read_configs()))
    total_configs_to_test = len(list(test_image_dir.glob("*.bmp"))) * len(list(generate_read_configs())) + total_configs_to_test
    total_configs_to_test = len(list(test_image_dir.glob("*.webp"))) * len(list(generate_read_configs())) + total_configs_to_test
    total_configs_to_test = len(list(test_image_dir.glob("*.tiff"))) * len(list(generate_read_configs())) + total_configs_to_test
    total_configs_to_test = len(list(test_image_dir.glob("*.jpg"))) * len(list(generate_read_configs())) + total_configs_to_test
    print(f"Testing {total_configs_to_test} configurations")
    exts = ["*.png",
    "*.bmp",
    "*.webp",
    "*.tiff",
    "*.jpg"]
    tq = tqdm(total=total_configs_to_test, desc="Testing configurations")
    for ext in exts:
        for image_path in test_image_dir.glob(ext):
            for config in generate_read_configs():
                test_read_image(image_path, config)
                tq.update(1)