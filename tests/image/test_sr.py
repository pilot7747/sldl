import pytest

from os.path import dirname, abspath, join
from shutil import rmtree
from PIL import Image
from sldl.image import ImageSR
from sldl._utils import get_data_dir


@pytest.mark.parametrize('model_name', ['SwinIR-M', 'SwinIR-L', 'BSRGAN', 'BSRGANx2'])
def test_sr(model_name):
    img_path = join(dirname(dirname(abspath(__file__))), 'test_files', 'image.jpg')
    sr = ImageSR(model_name)
    img = Image.open(img_path)
    sr(img)

    # gt_path = join(dirname(dirname(abspath(__file__))), 'test_files', f'image_{model_name}.png')
    # gt = Image.open(gt_path)
    # if model_name not in ['SwinIR-M', 'SwinIR-L']:
    #     assert_allclose(np.asarray(upscaled) / 255., np.asarray(gt) / 255.)
    rmtree(get_data_dir())
