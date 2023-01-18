from torch import nn
from tqdm.auto import tqdm


from sldl.image import ImageDenoising
from sldl._utils import get_video_frames, frames_to_video, get_fps


class VideoDenoising(nn.Module):
    r"""Video Denoising

    Takes a noisy video and removes the noise from it. Currently supports only
    `SwinIR` that is applied to a video frame-by-frame.

    :param model_name: Name of the pre-trained model. Now it can only be `SwinIR`.
    :type model_name: str
    :param noise: Noise level that the model was trained on. Can be of of
        `15`, `25`, `50`.
    :type noise: int
    :param precision:  Can be either `full` (uses fp32) and `half` (uses fp16).
        Default: `full`.
    :type precision: str.

    Example:

    .. code-block:: python

        from sldl.video import VideoDenoising

        denoiser = VideoDenoising('BSRGAN').cuda()
        sr('your_video.mp4', 'denoised_video.mp4')
    """

    def __init__(self, model_name="SwinIR", noise=15, precision: str = "full"):
        super(VideoDenoising, self).__init__()
        self.model_name = model_name
        self.precision = precision

        if model_name == "SwinIR":
            self.model = ImageDenoising(model_name, noise=noise, precision=precision)

    def _apply_swinir(self, path):
        frames = get_video_frames(path)
        return [self.model(frame) for frame in tqdm(frames)]

    def __call__(self, path, dest):
        """Denoises the image

        :param path: Path to the source video file.
        :type path: str
        :param dest: Path where the denoised version should be saved.
        :type dest: str
        """
        if self.model_name == "SwinIR":
            out_frames = self._apply_swinir(path)
        fps = get_fps(path)
        frames_to_video(out_frames, dest, fps)
