import torch
from torch import nn
from .swinir import SwinIR, swin_ir_inference
from PIL import Image

from sldl._utils import get_checkpoint_path


class ImageDenoising(nn.Module):
    r"""Image Denoising

    Takes an image and removes the noise from it. Currently supports
    SwinIR only.

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
    
        from PIL import Image
        from sldl.image import ImageDenoising

        denoiser = ImageDenoising('SwinIR')
        img = Image.open('test.png')
        denoised = denoiser(img)
    """
    def __init__(self, model_name: str = 'SwinIR', noise: int = 15, precision: str = 'full'):
        super(ImageDenoising, self).__init__()
        self.model_name = model_name
        self.precision = precision
        if model_name == 'SwinIR':
            self.model = SwinIR(upscale=1, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
            path = get_checkpoint_path(f'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise{noise}.pth')
            pretrained_model = torch.load(path)
            self.model.load_state_dict(pretrained_model['params'] if 'params' in pretrained_model.keys() else pretrained_model, strict=True)
        if self.precision == 'half':
            self.model = self.model.half()
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Applies the denoiser.

        Args:
            img (PIL.Image.Image): An input image.
        
        Returns:
            PIL.Image.Image: Denoised image.
        """
        return swin_ir_inference(self.model, img, device=self.device, precision=self.precision)
