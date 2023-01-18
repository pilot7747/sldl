import pytest

from os.path import dirname, abspath, join
from shutil import rmtree
from PIL import Image
from sldl.image import ImageDenoising
from sldl._utils import get_data_dir


@pytest.mark.parametrize('model_name', ['SwinIR'])
def test_denoising(model_name):
    img_path = join(dirname(dirname(abspath(__file__))), 'test_files', 'image.jpg')
    img = Image.open(img_path)
    denoiser = ImageDenoising(model_name)
    denoiser(img)
    rmtree(get_data_dir())
