# Single-Line Deep Learning

Most of the practicle tasks that require the usage of deep learning models can be simplified to "just do the thing", e.g., "just upscale the image". On the other hand, official repositories of the state-of-the-art methods are dedicated to provide a way to reproduce experiments presented in the paper. These two different tasks require different code structure, so I made this library that provides an ultimative single-line solution for practical tasks solved by SOTA methods. For instance, to "just upscale the image" you can just run the following code:

```python
from PIL import Image
from sldl.image import ImageSR

sr = ImageSR('BSRGAN')

img = Image.open('test.png')
upscaled = sr(img)
```
