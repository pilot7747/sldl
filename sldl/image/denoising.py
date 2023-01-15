import torch
from torch import nn
from .swinir import SwinIR, swin_ir_inference

from sldl.utils import get_checkpoint_path


class ImageDenoising(nn.Module):
    def __init__(self, model_name='SwinIR', noise=15):
        super(ImageDenoising, self).__init__()
        self.model_name = model_name
        if model_name == 'SwinIR':
            self.model = SwinIR(upscale=1, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
            path = get_checkpoint_path(f'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise{noise}.pth')
            pretrained_model = torch.load(path)
            self.model.load_state_dict(pretrained_model['params'] if 'params' in pretrained_model.keys() else pretrained_model, strict=True)
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def __call__(self, img):
        return swin_ir_inference(self.model, img, device=self.device)
