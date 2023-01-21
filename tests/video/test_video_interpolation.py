import pytest

import os
from os.path import dirname, abspath, join
from shutil import rmtree
from sldl.video import VideoInterpolation
from sldl._utils import get_data_dir


@pytest.mark.parametrize("model_name", ["IFRNet-Vimeo"])
def test_video_interpolation(model_name):
    video_path = join(dirname(dirname(abspath(__file__))), "test_files", "video.mp4")
    interpolator = VideoInterpolation(model_name)
    interpolator(video_path, "interpolated.mp4")

    os.remove("interpolated.mp4")
    rmtree(get_data_dir())
