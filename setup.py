from setuptools import setup, find_packages

setup(
    name="functis",
    version="0.1",
    packages=find_packages(),
    requires=[
        "numpy",
        "torch",
        "opencv_python_headless",
        "pillow",
        "torchvision",
        "pydantic",
        "pyyaml",
    ],
    author="Alex Redden",
    author_email="alex@aredden.net",
    description="Universal image reader designed for seamless reading of images into multiple formats, including PIL Images, Torch Tensors, and NumPy arrays using OpenCV",
    long_description=open("README.md").read(),
    url="https://github.com/aredden/functis",
    classifiers=[
        # Classifiers help users find your project by categorizing it.
        # For a list of valid classifiers, see https://pypi.org/classifiers/
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11"
    ],
)