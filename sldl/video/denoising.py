from torch import nn
from tqdm.auto import tqdm


from sldl.image import ImageDenoising
from sldl.utils import get_video_frames, frames_to_video, get_fps


class VideoDenoising(nn.Module):
    def __init__(self, model_name='SwinIR', noise=15):
        super(VideoDenoising, self).__init__()
        self.model_name = model_name
        
        if model_name == 'SwinIR':
            self.model = ImageDenoising(model_name, noise=noise)
            
    def apply_swinir(self, path):
        frames = get_video_frames(path)
        return [self.model(frame) for frame in tqdm(frames)]
        
    def __call__(self, path, dest):
        if self.model_name == 'SwinIR':
            out_frames = self.apply_swinir(path)
        fps = get_fps(path)
        frames_to_video(out_frames, dest, fps)
