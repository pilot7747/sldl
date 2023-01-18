import cv2
import numpy as np
import torch
from torch import nn
from PIL import Image
from tqdm.auto import tqdm
from typing import Optional, Tuple

from .vrt import VRT

from sldl.image import ImageSR
from sldl._utils import get_video_frames, frames_to_video, get_fps, get_checkpoint_path


class VideoSR(nn.Module):
    r"""Video Super-Resolution

    Takes an image and increases its resoulution by some factor. Currently supports
    SwinIR, BSRGAN, and VRT models.

    :param model_name: Name of the pre-trained model. Can be one of the `SwinIR-M`,
        `SwinIR-L`, `BSRGAN`, `BSRGANx2`, and `vrt`. Default: `BSRGAN`.
    :type model_name: str
    :param precision:  Can be either `full` (uses fp32) and `half` (uses fp16).
        Default: `full`.
    :type precision: str

    Example:

    .. code-block:: python

        from sldl.video import VideoSR

        sr = VideoSR('BSRGAN').cuda()
        sr('your_video.mp4', 'upscaled_video.mp4')
    """

    def __init__(self, model_name="BSRGAN", precision="full"):
        super(VideoSR, self).__init__()
        self.model_name = model_name
        self.precision = precision
        if model_name == "vrt":
            self.model = VRT(
                upscale=4,
                img_size=[6, 64, 64],
                window_size=[6, 8, 8],
                depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4],
                indep_reconsts=[11, 12],
                embed_dims=[
                    120,
                    120,
                    120,
                    120,
                    120,
                    120,
                    120,
                    180,
                    180,
                    180,
                    180,
                    180,
                    180,
                ],
                num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                pa_frames=2,
                deformable_groups=12,
            )
            path = get_checkpoint_path(
                "https://github.com/JingyunLiang/VRT/releases/download/v0.0/001_VRT_videosr_bi_REDS_6frames.pth"
            )
            pretrained_model = torch.load(path)
            self.model.load_state_dict(
                pretrained_model["params"]
                if "params" in pretrained_model.keys()
                else pretrained_model,
                strict=True,
            )
            self.tile = [40, 128, 128]
            if precision == "half":
                self.model = self.model.half()
        elif model_name in ["SwinIR-M", "SwinIR-L", "BSRGAN", "BSRGANx2"]:
            self.model = ImageSR(model_name, precision=precision)

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def _video_to_tensor(path):
        frames = get_video_frames(path)
        return (
            torch.stack(
                [
                    torch.from_numpy(np.asarray(img)[:, :, :3].transpose(2, 0, 1))
                    for img in frames
                ]
            ) / 255.0
        )

    def _test_clip(self, lq):
        sf = 4
        window_size = [6, 8, 8]
        size_patch_testing = self.tile[1]
        tile_overlap = [2, 20, 20]
        assert (
            size_patch_testing % window_size[-1] == 0
        ), "testing patch size should be a multiple of window_size."

        if size_patch_testing:
            # divide the clip to patches (spatially only, tested patch by patch)
            overlap_size = tile_overlap[1]
            not_overlap_border = True

            # test patch by patch
            b, d, c, h, w = lq.size()
            stride = size_patch_testing - overlap_size
            h_idx_list = list(range(0, h - size_patch_testing, stride)) + [
                max(0, h - size_patch_testing)
            ]
            w_idx_list = list(range(0, w - size_patch_testing, stride)) + [
                max(0, w - size_patch_testing)
            ]
            E = torch.zeros(b, d, c, h * sf, w * sf)
            W = torch.zeros_like(E)

            for h_idx in tqdm(h_idx_list):
                for w_idx in w_idx_list:
                    in_patch = lq[
                        ...,
                        h_idx:h_idx + size_patch_testing,
                        w_idx:w_idx + size_patch_testing,
                    ]
                    out_patch = self.model(in_patch).detach().cpu()

                    out_patch_mask = torch.ones_like(out_patch)

                    if not_overlap_border:
                        if h_idx < h_idx_list[-1]:
                            out_patch[..., -overlap_size // 2:, :] *= 0
                            out_patch_mask[..., -overlap_size // 2:, :] *= 0
                        if w_idx < w_idx_list[-1]:
                            out_patch[..., :, -overlap_size // 2:] *= 0
                            out_patch_mask[..., :, -overlap_size // 2:] *= 0
                        if h_idx > h_idx_list[0]:
                            out_patch[..., : overlap_size // 2, :] *= 0
                            out_patch_mask[..., : overlap_size // 2, :] *= 0
                        if w_idx > w_idx_list[0]:
                            out_patch[..., :, : overlap_size // 2] *= 0
                            out_patch_mask[..., :, : overlap_size // 2] *= 0

                    E[
                        ...,
                        h_idx * sf:(h_idx + size_patch_testing) * sf,
                        w_idx * sf:(w_idx + size_patch_testing) * sf,
                    ].add_(out_patch)
                    W[
                        ...,
                        h_idx * sf:(h_idx + size_patch_testing) * sf,
                        w_idx * sf:(w_idx + size_patch_testing) * sf,
                    ].add_(out_patch_mask)
            output = E.div_(W)
            return output

    @torch.no_grad()
    def _apply_vrt(self, path):
        lq = VideoSR._video_to_tensor(path).to(self.device)[None, :, :, :, :]
        if self.precision == "half":
            lq = lq.half()
        num_frame_testing = self.tile[0]
        tile_overlap = [2, 20, 20]

        sf = 4

        num_frame_overlapping = tile_overlap[0]
        not_overlap_border = False
        b, d, c, h, w = lq.size()
        stride = num_frame_testing - num_frame_overlapping
        d_idx_list = list(range(0, d - num_frame_testing, stride)) + [
            max(0, d - num_frame_testing)
        ]
        E = torch.zeros(b, d, c, h * sf, w * sf)
        W = torch.zeros(b, d, 1, 1, 1)

        for d_idx in tqdm(d_idx_list):
            lq_clip = lq[:, d_idx:d_idx + num_frame_testing, ...]
            out_clip = self._test_clip(lq_clip)
            out_clip_mask = torch.ones((b, min(num_frame_testing, d), 1, 1, 1))

            if not_overlap_border:
                if d_idx < d_idx_list[-1]:
                    out_clip[:, -num_frame_overlapping // 2:, ...] *= 0
                    out_clip_mask[:, -num_frame_overlapping // 2:, ...] *= 0
                if d_idx > d_idx_list[0]:
                    out_clip[:, : num_frame_overlapping // 2, ...] *= 0
                    out_clip_mask[:, : num_frame_overlapping // 2, ...] *= 0

            E[:, d_idx:d_idx + num_frame_testing, ...].add_(out_clip)
            W[:, d_idx:d_idx + num_frame_testing, ...].add_(out_clip_mask)
        output = E.div_(W)

        for i in range(output.shape[1]):
            # save image
            img = output[:, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
            if img.ndim == 3:
                img = np.transpose(
                    img[[2, 1, 0], :, :], (1, 2, 0)
                )  # CHW-RGB to HCW-BGR
            img = (img * 255.0).round().astype(np.uint8)
            yield Image.fromarray(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def _apply_imagesr(self, path, pre_resolution=None):
        frames = get_video_frames(path, pre_resolution)
        for frame in tqdm(frames):
            yield self.model(frame)
        # return [self.model(frame) for frame in tqdm(frames)]

    def __call__(
        self, path: str, dest: str, pre_resolution: Optional[Tuple[int, int]] = None
    ) -> None:
        """Upscales the image

        :param path: Path to the source video file.
        :type path: str
        :param dest: Path where the upscaled version should be saved.
        :type dest: str
        """
        fps = get_fps(path)
        if self.model_name == "vrt":
            out_frames = self._apply_vrt(path)
        elif self.model_name in ["SwinIR-M", "SwinIR-L", "BSRGAN", "BSRGANx2"]:
            out_frames = self._apply_imagesr(path, pre_resolution)
        frames_to_video(out_frames, dest, fps)
