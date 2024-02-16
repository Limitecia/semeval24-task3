from __future__ import annotations
from typing import List, Tuple, Dict
from data.conversation import Conversation
from data.relation import CauseRelation
from torch.utils.data import Dataset
import json, torch
from utils import flatten_list
from random import shuffle

class Subtask1Dataset(Dataset):
    INPUT_FIELDS = ['TEXT', 'SPEAKER']
    TARGET_FIELDS = ['EMOTION', 'GRAPH', 'SPAN']
    

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
        path: str
    ) -> Subtask1Dataset:
        with open(path, 'r') as reader:
            data = json.load(reader)

        convs = [Conversation.from_dict(d, subtask=1) for d in data]
        data = Subtask1Dataset(convs)
        return data

    def split(self, ptest: float) -> Tuple[Subtask1Dataset, Subtask1Dataset]:
        shuffle(self.convs)
        ntest = int(len(self)*ptest)
        test = self.convs[:ntest]
        train = self.convs[ntest:]
        train, test = [Subtask1Dataset(d) for d in (train, test)]
        return train, test

    def save(self, path: str, submission: bool = True):
        data = [conv.to_dict(submission) for conv in sorted(self.convs, key=lambda conv: conv.id)]
        with open(path, 'w') as writer:
            json.dump(data, writer, indent=4)

            
    @property
    def lens(self) -> torch.Tensor:
        return torch.Tensor([sum(conv.lens) for conv in self.convs])
    
    
    def join(self, other: Subtask1Dataset) -> Subtask1Dataset:
        self.convs += other.convs
        return self
        