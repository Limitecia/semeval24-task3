from typing import Tuple, List, Optional
import torch, time, os, cv2
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoImageProcessor
from transformers.feature_extraction_utils import BatchFeature
import numpy as np 
from utils import flatten_list, split 
from dataclasses import dataclass
from torchvision.transforms import Resize
from torch.nn.functional import interpolate
from tqdm import tqdm 
from concurrent.futures import ThreadPoolExecutor


def read_videos(paths: List[str], num_frames: int) -> List[torch.Tensor]:
    return [read_video(path, num_frames) for path in tqdm(paths, total=len(paths), desc='FRAME')]

def read_video(path: str, num_frames: int) -> torch.Tensor:
    assert os.path.exists(path), f'Video {path} does not exist'
    video = cv2.VideoCapture(path)
    res, frame = video.read()
    frames = []
    while res:
        frames.append(torch.tensor(frame))
        res, frame = video.read()
    frames = torch.stack(frames, 0)
    if frames.shape[0] > (num_frames + 2):
        indices = [i for i in range(frames.shape[0]) if (i+1) % (frames.shape[0]//(num_frames+2)) == 0]
        frames = frames[indices[1:-1]]
    else:
        frames = frames[:num_frames]
    return frames


class ImageProcessor:
    EXTENSION = 'img'
    OBJECTS = ['field', 'pretrained', 'num_frames']
    TRAINABLE = False

    def __init__(
        self, 
        field: str, 
        pretrained: str,
        num_frames: int = 5,
        num_workers: int = 12, 
    ):
        self.field = field
        self.pretrained = pretrained
        self.num_frames = num_frames
        self.num_workers = num_workers
        
        # image processoor
        self.processor = AutoImageProcessor.from_pretrained(pretrained)
        self.data = dict()

        for key in self.OBJECTS:
            self.__setattr__(key, locals()[key])

    def fit(self, paths: List[str]):
        # with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            # batch_size = len(paths)//self.num_workers
            # futures = [pool.submit(read_videos, paths[i:(i+batch_size)], self.num_frames) for i in range(0, len(paths), batch_size)]
            # futures = flatten_list(f.result() for f in futures)
            # futures = list(tqdm(pool.map(read_video, paths, [self.num_frames for _ in paths]), total=len(paths), desc=self.field))
        # self.data = dict(zip(paths, futures))
        pass

    def encode(self, paths: List[str]) -> List[torch.Tensor]:
        return [self.processor(self.data[path], return_tensors='pt').pixel_values for path in paths]


    def batch_encode(self, batch: List[List[str]]) -> List[List[torch.Tensor]]:
        nonstored = list(set(flatten_list(batch)) - self.data.keys())
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            videos = pool.map(read_video, nonstored, [self.num_frames for _ in nonstored])
        videos = [self.processor(video, return_tensors='pt').pixel_values for video in videos]
        self.data.update(**dict(zip(nonstored, videos)))
        return [[self.data[path] for path in paths] for paths in batch]
        