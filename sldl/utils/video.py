import cv2
import os
import sys
from tqdm.auto import tqdm
from PIL import Image
import numpy as np


def get_video_frames(path):
    cam = cv2.VideoCapture(path)
    currentframe = 0
    frames = []
    while(True):
        ret, frame = cam.read()

        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)).convert('RGB'))
            currentframe += 1
        else:
            break
    cam.release()
    return frames


def frames_to_video(frames, path, fps):
        width, height = frames[0].size
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video = cv2.VideoWriter(path,  fourcc, fps, (width, height))

        for image in frames:
            video.write(np.asarray(image.convert('RGB'))[:, :, ::-1].copy())
        cv2.destroyAllWindows()
        video.release()
        

def get_fps(path):
        vidcap = cv2.VideoCapture(path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        vidcap.release()
        return fps
