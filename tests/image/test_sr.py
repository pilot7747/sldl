import pytest

from os.path import dirname, abspath, join
import numpy as np
from numpy.testing import assert_allclose
from PIL import Image
from sldl.image import ImageSR


@pytest.mark.parametrize('model_name', ['SwinIR-M', 'SwinIR-L', 'BSRGAN', 'BSRGANx2'])
def test_sr(model_name):
    img_path = join(dirname(dirname(abspath(__file__))), 'test_files', 'image.jpg')
    sr = ImageSR(model_name)
    img = Image.open(img_path)
    upscaled = sr(img)

    gt_path = join(dirname(dirname(abspath(__file__))), 'test_files', f'image_{model_name}.png')
    gt = Image.open(gt_path)
    if model_name not in ['SwinIR-M', 'SwinIR-L']:
        assert_allclose(np.asarray(upscaled) / 255., np.asarray(gt) / 255.)
