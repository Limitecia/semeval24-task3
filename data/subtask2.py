from __future__ import annotations
from typing import List, Tuple, Dict
from data.conversation import Conversation
from torch.utils.data import Dataset
import json, torch, os 
from utils import flatten_list, Tokenizer
from random import shuffle
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm 

class Subtask2Dataset(Dataset):
    
    INPUT_FIELDS = ['TEXT', 'SPEAKER', 'FRAME', 'AUDIO']
    TARGET_FIELDS = ['EMOTION', 'GRAPH']

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
    
        
    @classmethod
    def from_path(
        cls,
        path: str,
        folder: str,
        num_workers: int = 20,
    ) -> Subtask2Dataset:
        with open(path, 'r') as reader:
            data = json.load(reader)
        convs = [Conversation.from_dict(d,2,folder) for d in data]
        data = Subtask2Dataset(convs)
        return data

    def split(self, ptest: float) -> Tuple[Subtask2Dataset, Subtask2Dataset]:
        shuffle(self.convs)
        ntest = int(len(self)*ptest)
        test = self.convs[:ntest]
        train = self.convs[ntest:]
        train, test = [Subtask2Dataset(d) for d in (train, test)]
        return train, test

    def save(self, path: str, submission: bool = True):
        data = [conv.to_dict(submission) for conv in sorted(self.convs, key=lambda conv: conv.id)]
        with open(path, 'w') as writer:
            json.dump(data, writer, indent=4)

            
    @property
    def lens(self) -> torch.Tensor:
        return torch.Tensor([sum(conv.lens) for conv in self.convs])
    
    
    def join(self, other: Subtask2Dataset) -> Subtask2Dataset:
        self.convs += other.convs
        return self
        
