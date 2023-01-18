import pytest

import os
from os.path import dirname, abspath, join
from shutil import rmtree
from PIL import Image
from sldl.video import VideoSR
from sldl._utils import get_data_dir


@pytest.mark.parametrize('model_name', ['vrt', 'BSRGAN'])
def test_video_sr(model_name):
    video_path = join(dirname(dirname(abspath(__file__))), 'test_files', 'video.mp4')
    sr = VideoSR(model_name)
    sr(video_path, 'sr.mp4')

    os.remove('sr.mp4')
    rmtree(get_data_dir())
