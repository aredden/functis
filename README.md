# Functis: A Universal Image Reader

`functis` is a universal image reader designed for seamless reading of images into multiple formats, including PIL Images, torch Tensors via torchvision.io, and NumPy arrays via OpenCV. It offers flexibility in device placement, image format, color modes, data types, and output types.

## Features

- **Read Modes**: Supports multiple reading backends including `torch`, `np`, `pil`, and `nvjpeg`.
    > note: torch is torchvision.io, np uses cv2, and nvjpeg is a private library I have yet to expose publically.
- **Device Placement**: Place your data on any CUDA device or CPU.
- **Image Format**: Choose between `hwc` or `chw` formats.
- **Color Modes**: Multiple color options including `rgb`, `bgr`, `rgba`, `bgra`, and `grayscale`.
- **Data Types**: Supports `uint8`, `float32`, and `float16` for `torch`, `nvjpeg` and `np` backends.
- **Output Types**: Get the result in `pil`, `np`, or `torch` formats.

## Installation

```bash
git clone https://github.com/aredden/functis
cd functis
python -m pip install .
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

# For recursive reading...
for image in imreader.read_dir("./your_image_directory", recursive=True):
    # process or view your image
    ...

# Also will optionally include paths...
for path, image in imreader.read_dir("./your_image_directory", return_paths=True):
    # process or view your image
    ...
```

### Reading a List of Images

> Can also return paths, similar to read_dir.
```python
for image in imreader.read_list(list_of_your_images):
    # process or view your image
    ...
```

### Reading images as bytes

> Relatively new feature (as of oct 11 2023)

```python
with open("./your_image.jpg", "rb") as f:
    image = imreader.read(f.read())
```

Inspiration for this idea is that I wanted to have a way to efficienctly read images of all types from a web api, here is a simple fastapi example:

```python
from typing import Annotated
from fastapi import FastAPI, Body

app = FastAPI()
inference_client = ...

@app.post("/inference", tags=["inference"])
async def inference(
    image_bytes: Annotated[bytes, Body(media_type="application/octet-stream")]
):
    global inference_client
    image = imreader.read(image_bytes)

    with torch.no_grad():
        results_json = inference_client.infer(image)
    return results_json
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
