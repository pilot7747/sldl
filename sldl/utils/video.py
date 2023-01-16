import cv2
import os
import sys
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
from typing import Any, Iterable, List, Optional, Tuple


def get_video_frames(path: str, resolution: Optional[Tuple[int, int]] = None) -> List[Any]:
    cam = cv2.VideoCapture(path)
    currentframe = 0
    frames = []
    while(True):
        ret, frame = cam.read()

        if ret:
            if resolution is None:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)).convert('RGB'))
            else:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)).convert('RGB').resize(resolution))
            currentframe += 1
        else:
            break
    cam.release()
    return frames


def frames_to_video(frames: Iterable[Any], path: str, fps: float) -> None:
        first_frame = next(frames)
        width, height = first_frame.size
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video = cv2.VideoWriter(path,  fourcc, fps, (width, height))
        video.write(np.asarray(first_frame.convert('RGB'))[:, :, ::-1].copy())

        for image in frames:
            video.write(np.asarray(image.convert('RGB'))[:, :, ::-1].copy())
        cv2.destroyAllWindows()
        video.release()


def get_fps(path: str) -> float:
        vidcap = cv2.VideoCapture(path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        vidcap.release()
        return fps
