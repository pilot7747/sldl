import pytest

import os
from os.path import dirname, abspath, join
from shutil import rmtree
from PIL import Image
from sldl.video import VideoDenoising
from sldl._utils import get_data_dir


@pytest.mark.parametrize('model_name', ['SwinIR'])
def test_video_denoising(model_name):
    video_path = join(dirname(dirname(abspath(__file__))), 'test_files', 'video.mp4')
    denoiser = VideoDenoising(model_name)
    denoiser(video_path, 'denoised.mp4')

    os.remove('denoised.mp4')
    rmtree(get_data_dir())
