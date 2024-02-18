from __future__ import annotations
from typing import List, Tuple, Dict
from data.conversation import Conversation
from torch.utils.data import Dataset
import json, torch
from utils import flatten_list
from random import shuffle

class SubtaskDataset(Dataset):
    INPUT_FIELDS, TARGET_FIELDS = None, None 
    
    def __init__(self, convs: List[Conversation]):
        super().__init__()
        self.convs = convs

    @property
    def max_len(self):
        return max(map(len, self.convs))
    
    @property
    def avg_span_len(self):
        lens = flatten_list(conv.span_lens for conv in self.convs)
        return sum(lens)/len(lens)

    def __len__(self):
        return len(self.convs)
    
    
    def __getitem__(self, index: int) -> Tuple[list, list, Conversation]:
        conv = self.convs[index]
        inputs = [getattr(conv, field) for field in self.INPUT_FIELDS]
        targets = [getattr(conv, field) for field in self.TARGET_FIELDS]
        return inputs, targets, conv
    
    
    def split(self, ptest: float) -> Tuple[SubtaskDataset, SubtaskDataset]:
        shuffle(self.convs)
        ntest = int(len(self)*ptest)
        test = self.convs[:ntest]
        train = self.convs[ntest:]
        return self.__class__(train), self.__class__(test)

    def save(self, path: str, submission: bool = True):
        data = [conv.to_dict(submission) for conv in sorted(self.convs, key=lambda conv: conv.id)]
        with open(path, 'w') as writer:
            json.dump(data, writer, indent=4)
            
    
            
    @property
    def lens(self) -> torch.Tensor:
        return torch.Tensor([sum(conv.lens) for conv in self.convs])
    

class Subtask1Dataset(SubtaskDataset):
    INPUT_FIELDS = ['TEXT', 'SPEAKER']
    TARGET_FIELDS = ['EMOTION', 'GRAPH', 'SPAN']
    
    @classmethod
    def from_path(cls, path: str) -> Subtask1Dataset:
        with open(path, 'r') as reader:
            data = json.load(reader)
        convs = [Conversation.from_dict(d) for d in data]
        data = cls(convs)
        return data


class Subtask2Dataset(SubtaskDataset):
    INPUT_FIELDS = ['TEXT', 'SPEAKER', 'FRAME', 'AUDIO']
    TARGET_FIELDS = ['EMOTION', 'GRAPH']
    
    @classmethod
    def from_path(cls, path: str, video_folder: str) -> Subtask2Dataset:
        with open(path, 'r') as reader:
            data = json.load(reader)
        convs = [Conversation.from_dict(d, video_folder) for d in data]
        data = cls(convs)
        return data



        
