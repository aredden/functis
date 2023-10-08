import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.as_posix()))
parent_dir = Path(__file__).parent.parent

import cv2
import numpy as np

extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def generate_image(channels: int = 3, dtype: np.dtype = np.uint8):
    if channels == 1:
        return np.random.randint(0, 255, (256, 256), dtype=dtype)
    else:
        return np.random.randint(0, 255, (256, 256, channels), dtype=dtype)


channel_options_per_extension = {
    ".jpg": [1, 3, 4],
    ".jpeg": [1, 3, 4],
    ".png": [1, 3, 4],
    ".bmp": [1, 3, 4],
    ".tif": [1, 3, 4],
    ".tiff": [1, 3, 4],
    ".webp": [1, 3, 4],
}

dtype_options_per_extension = {
    ".jpg": [np.uint8],
    ".jpeg": [np.uint8],
    ".png": [np.uint8],
    ".bmp": [np.uint8],
    ".tif": [np.uint8],
    ".tiff": [np.uint8],
    ".webp": [np.uint8],
}


def generate_images():
    for ext in extensions:
        for channels in channel_options_per_extension[ext]:
            for dtype in dtype_options_per_extension[ext]:
                name = f"test_image_{channels}c_{dtype.__name__}.{ext}".replace(
                    "..", "."
                )
                yield generate_image(channels=channels, dtype=dtype), name


def write_images():
    for img, name in generate_images():
        cv2.imwrite((parent_dir / "tests" / "test_images" / name).as_posix(), img)


if __name__ == "__main__":
    write_images()
