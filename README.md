# Functis: A Universal Image Reader

`functis` is a universal image reader designed for seamless reading of images into multiple formats, including PIL Images, Torch Tensors, and NumPy arrays using OpenCV (cv2). It offers flexibility in device placement, image format, color modes, data types, and output types.

## Features

- **Read Modes**: Supports multiple reading backends including `torch`, `np`, `pil`, and `nvjpeg`.
- **Device Placement**: Place your data on any CUDA device or CPU.
- **Image Format**: Choose between `hwc` or `chw` formats.
- **Color Modes**: Multiple color options including `rgb`, `bgr`, `rgba`, `bgra`, and `grayscale`.
- **Data Types**: Supports `uint8`, `float32`, and `float16` for `torch` and `np` backends.
- **Output Types**: Get the result in `pil`, `np`, or `torch` formats.

## Installation

```bash
pip install functis
```

## Usage

### Basic Usage

```python
from functis import ImageReader

imreader = ImageReader().with_reader("torch").with_device("cuda:0").with_format("hwc")
image = imreader.read("./your_image.jpg")
```

### Reading Multiple Images from a Directory

```python
for image in imreader.read_dir("./your_image_directory"):
    # process or view your image
    ...
```

### Reading a List of Images

```python
for image in imreader.read_images(list_of_your_images):
    # process or view your image
    ...
```

### Options

- **Read Modes**: `torch`, `np`, `pil`, `nvjpeg` # torch nvjpeg reader is not public yet.
- **Device**: `cuda:<index>`, `cuda`, `cpu`
- **Format**: `hwc`, `chw`
- **Color Mode**: `rgb`, `bgr`, `rgba`, `bgra`, `grayscale`
- **Data Type**: `uint8`, `float32`, `float16` (only for torch and np)
- **Image Type**: `pil`, `np`, `torch`

## Contributing

We welcome contributions! Please see our [contributing guidelines](LINK_TO_CONTRIBUTING.md) for more details.
