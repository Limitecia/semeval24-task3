from transformers import AutoFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from typing import List 
from torchaudio.functional import resample
from utils import flatten_list, split
import numpy as np 
import torch

class AudioProcessor:
    EXTENSION = 'audio'
    OBJECTS = ['field', 'pretrained']
    TRAINABLE = False 
    
    def __init__(self, field: str, pretrained: str, audio_rate: int = 48000):
        self.field = field 
        self.pretrained = pretrained 
        self.processor = AutoFeatureExtractor.from_pretrained(pretrained)
        self.audio_rate = audio_rate
        self.args = dict(return_tensors='pt', padding=True, sampling_rate=self.processor.sampling_rate, return_attention_mask=True)
        
        
    def batch_encode(self, batch: List[List[str]]) -> List[BatchFeature]:
        lens = list(map(len, batch))
        audios = [torch.load(audio).sum(0)/2 for audio in flatten_list(batch)]
        audios = split([resample(audio, self.audio_rate, self.processor.sampling_rate).numpy() for audio in audios], lens)
        audios = [self.processor(audio, **self.args) for audio in audios]
        return audios