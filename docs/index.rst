.. SLDL documentation master file, created by
   sphinx-quickstart on Tue Jan 17 13:12:24 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Single-Line Deep Learning Documentation
=======================================

Most of the practicle tasks that require the usage of deep learning models can be simplified to 
"just do the thing", e.g., "just upscale the image". On the other hand, official repositories of the state-of-the-art methods 
are dedicated to provide a way to reproduce experiments presented in the paper. These two different tasks require different code structure, 
so I made this library that provides an ultimative single-line solution for practical tasks solved by the SOTA methods.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Installation
============

You can install SLDL by running the following command:

.. code-block:: bash

   pip install sldl


Supported Models
================

.. list-table:: Image Super-Resolution
   :widths: 1 1 1 1 1 1
   :header-rows: 1

   * - 
     - `SwinIR-M <https://github.com/JingyunLiang/SwinIR>`_
     - `SwinIR-L <https://github.com/JingyunLiang/SwinIR>`_
     - `BSRGAN <https://github.com/cszn/BSRGAN>`_
     - `BSRGANx2 <https://github.com/cszn/BSRGAN>`_
     - `RealESRGAN <https://github.com/xinntao/Real-ESRGAN>`_
   * - Upscale factor
     - 4
     - 4
     - 4
     - 2
     - 4

.. list-table:: Video Super-Resolution
   :widths: 1 1 1 1 1 1 1
   :header-rows: 1

   * - 
     - `SwinIR-M <https://github.com/JingyunLiang/SwinIR>`_
     - `SwinIR-L <https://github.com/JingyunLiang/SwinIR>`_
     - `BSRGAN <https://github.com/cszn/BSRGAN>`_
     - `BSRGANx2 <https://github.com/cszn/BSRGAN>`_
     - `RealESRGAN <https://github.com/xinntao/Real-ESRGAN>`_
     - `VRT <https://github.com/JingyunLiang/VRT>`_
   * - Upscale factor
     - 4
     - 4
     - 4
     - 2
     - 4
     - 4

* Image Denoising: `SwinIR <https://github.com/JingyunLiang/SwinIR>`_
* Video Denoising: `SwinIR <https://github.com/JingyunLiang/SwinIR>`_
* Video Interpolation `IFRNet <https://github.com/ltkong218/IFRNet>`_

Quick Start
===========

Here is a quick example of how to upscale an image using BSRGAN:

.. code-block:: python

   from PIL import image
   from sldl.image import ImageSR

   sr = ImageSR('BSRGAN')
   img = Image.open('test.png')
   upscaled = sr(img)

If you have a GPU, you can do it faster by moving the module on it:

.. code-block:: python

   sr = sr.cuda()

SLDL also supports an easy way to do everything in fp16:

.. code-block:: python

   sr = ImageSR('BSRGAN', precision='half')

You can easily apply the same model to upscale a video:

.. code-block:: python

   from sldl.video import VideoSR

   sr = VideoSR('BSRGAN', precision='half').cuda()
   sr('/path/to/your_video.mp4', '/path/to/upscaled_video.mp4')

Reference
=========

.. toctree::
   :maxdepth: 1

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
