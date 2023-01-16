# Single-Line Deep Learning

Most of the practicle tasks that require the usage of deep learning models can be simplified to "just do the thing", e.g., "just upscale the image". On the other hand, official repositories of the state-of-the-art methods are dedicated to provide a way to reproduce experiments presented in the paper. These two different tasks require different code structure, so I made this library that provides an ultimative single-line solution for practical tasks solved by the SOTA methods. For instance, to "just upscale the image" you can just run the following code:

```python
from PIL import Image
from sldl.image import ImageSR

sr = ImageSR('BSRGAN')

img = Image.open('test.png')
upscaled = sr(img)
```

## Overview

SLDL is written in PyTorch and tries not to change the original author's implementation and, at the same time, provide the fastest inference and the most convinient interface. Note that SLDL doesn't provide any interface to train or fine-tune the models.

Each method is a `torch.nn.Module` that has a `__call__` method that solves your task. The library tries to provide the most practical interface, so it operates with Pillow images and video files in order to use the upscaler in your program and to just upscale your video.

Currently two types of tasks are supported.

### Images

* Denoising: SwinIR
* Super-resolution: BSRGAN, SwinIR

### Videos

* Denoising: SwinIR
* Super-resolution: BSRGAN, SwinIR, VRT

## Usage

For images, run this

```python
from PIL import Image
from sldl.image import ImageSR

img = Image.open('test.png')
sr = ImageSR('BSRGAN')  # or 'SwinIR-M', 'SwinIR-L', 'BSRGANx2'
# sr = sr.cuda() if you have a GPU

upscaled = sr(img)
```

For videos, run this
```python
from sldl.video import VideoSR

sr = VideoSR('BSRGAN')
sr('your_video.mp4', 'upscaled_video.mp4')
```

## Plans

* Make this a usable Python package
* Prettify the code, write the docs
* Add image deblurring, face generation, machine translation, etc
* Add more models like RealESRGAN
* Make inference optimizations like `torch.compile` and TensorRT
* CLI tool and Docker image
* Ready-to-go REST API deployment
