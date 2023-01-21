import torch
from torch import nn
from tqdm.auto import trange
from huggingface_hub import hf_hub_url
import logging

from sldl._utils import get_video_frames, frames_to_video, get_fps, get_checkpoint_path

from .ifrnet import IFRNet, ifnet_inference


class VideoInterpolation(nn.Module):
    r"""Video Interpolation

    Takes an image and increases the FPS. Currently supports only IFRNet trained on Vimeo90K
    an GoPro datasets and only x2 FPS increasing.

    :param model_name: Name of the pre-trained model. Can be one of the `IFRNet-Vimeo`
    and `IFRNet-GoPro`. Default: `IFRNet-Vimeo`.
    :type model_name: str

    Example:

    .. code-block:: python

        from sldl.video import VideoInterpolation

        interpolator = VideoInterpolation('IFRNet-Vimeo').cuda()
        interpolator('your_video.mp4', 'interpolated_video.mp4')
    """

    def __init__(self, model_name: str = "IFRNet-Vimeo"):
        super(VideoInterpolation, self).__init__()
        self.model_name = model_name
        if model_name in ["IFRNet-Vimeo", "IFRNet-GoPro"]:
            self.model = IFRNet().eval()
            if model_name == "IFRNet-Vimeo":
                url = hf_hub_url("pavlichenko/ifrnet_vimeo", "IFRNet_Vimeo90K.pth")
            elif model_name == "IFRNet-GoPro":
                url = hf_hub_url("pavlichenko/ifrnet_gopro", "IFRNet_GoPro.pth")
            path = get_checkpoint_path(url)
            self.model.load_state_dict(torch.load(path))
        self._device = None

    @property
    def device(self):
        return next(self.parameters()).device

    def _apply_ifrnet(self, path, device=None):
        device = self.device if device is None else device
        frames = get_video_frames(path)
        for i in trange(len(frames) - 1):
            pred_frame = ifnet_inference(self.model, frames[i], frames[i + 1], device)
            yield frames[i]
            yield pred_frame
        yield frames[-1]

    @torch.no_grad()
    def __call__(self, source: str, dest: str) -> None:
        """Interpolates the video

        :param source: Path to the source video file.
        :type source: str
        :param dest: Path where the upscaled version should be saved.
        :type dest: str
        """
        device = self._device if self._device is not None else self.device
        self._device = device
        try:
            self.model = torch.jit.optimize_for_inference(
                torch.jit.script(self.model.eval())
            )
        except Exception:
            logging.warning("Skipping JIT optimization")
        fps = get_fps(source)
        out_frames = self._apply_ifrnet(source, device)
        frames_to_video(out_frames, dest, fps * 2)
