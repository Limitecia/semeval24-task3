from typing import Tuple, List
import torch
from torchvision.io import read_video
from torch.nn.utils.rnn import pad_sequence
from transformers import TvltFeatureExtractor, TvltImageProcessor
from transformers.feature_extraction_utils import BatchFeature
import numpy as np 
from utils import flatten_list
from dataclasses import dataclass

class VideoProcessor:
    EXTENSION = 'proc'
    OBJECTS = ['field', 'num_frames', 'image_patch', 'audio_patch', 'audio_rate', 'shortest_edge', 'spectrogram_length']
    TRAINABLE = False

    def __init__(
        self, 
        field: str, 
        num_frames: int = 12,
        image_patch: Tuple[int, int] = (64, 64),
        audio_patch: Tuple[int, int] = (64, 64),
        audio_rate: int = 48000,
        shortest_edge: int = 360,
        spectrogram_length: int = 1024
    ):
        self.field = field
        self.num_frames = num_frames
        
        
        # immage processor 
        self.img = TvltImageProcessor(
            size={'shortest_edge': shortest_edge}, patch_size=image_patch, padding=True, 
            padding_value=0, num_frames=num_frames
        )
        
        # audio processor
        self.audio_rate = audio_rate
        self.audio = TvltFeatureExtractor(
            spectrogram_length=spectrogram_length, patch_size=audio_patch, sampling_rate=audio_rate,
            padding=True, padding_value=0
        )
        
        for key in self.OBJECTS:
            self.__setattr__(key, locals()[key])
        

    def _encode(self, path: str) -> Tuple[torch.Tensor, np.ndarray]:
        frames, audio, _ = read_video(path, pts_unit='sec')
        if frames.shape[0] > self.num_frames:
            indices = [i for i in range(frames.shape[0]) if i % (frames.shape[0]//self.num_frames) == 0][:self.num_frames]
            frames = frames[indices]
        return frames, (audio.sum(0)/2).numpy()
        

    def encode(self, paths: List[str]) -> Tuple[List[torch.Tensor], List[np.ndarray]]:
        frames, audio = zip(*map(self._encode, paths))
        return frames, audio 

    def batch_encode(self, batch: List[List[str]]) -> Tuple[BatchFeature, BatchFeature]:
        frames, audio = map(flatten_list, zip(*map(self.encode, batch)))
        frames = pad_sequence(frames, batch_first=True, padding_value=0)
        return self.img.preprocess(frames, return_tensors='pt'), self.audio(audio, return_tensors='pt', sampling_rate=self.audio_rate, padding=True)




