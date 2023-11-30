from __future__ import annotations
from typing import List, Tuple
from data.conversation import Conversation
from torch.utils.data import Dataset
import json, torch, os
from utils import Tokenizer, parallel, PretrainedTokenizer, NullTokenizer, flatten_list


class Subtask1Dataset(Dataset):

    def __init__(
        self,
        conversations: List[Conversation],
        input_fields: List[str],
        target_fields: List[str]
    ):
        super().__init__()
        self.conversations = conversations
        self.input_fields = input_fields
        self.target_fields = target_fields

    @property
    def max_len(self):
        return max(map(len, self.conversations))

    @property
    def avg_span_len(self):
        lens = flatten_list(conv.span_lens for conv in self.conversations)
        return sum(lens)/len(lens)

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, index: int) -> Tuple[list, list]:
        conver = self.conversations[index]
        inputs = [getattr(conver, field) for field in self.input_fields]
        targets = [getattr(conver, field) for field in self.target_fields]
        return inputs, targets

    @classmethod
    def from_path(
        cls,
        path: str,
        inputs_fields: List[str],
        target_fields: List[str],
        build: bool,
        num_workers: int = os.cpu_count(),
        show: bool = True
    ) -> Subtask1Dataset:
        with open(path, 'r') as reader:
            data = json.load(reader)

        conversations = [Conversation.from_dict(d) for d in data]
        data = Subtask1Dataset(conversations, inputs_fields, target_fields)
        return data

    def split(self, pval: float, ptest: float) -> Tuple[Subtask1Dataset, Subtask1Dataset, Subtask1Dataset]:
        nval, ntest = int(len(self)*pval), int(len(self)*ptest)
        ntrain = len(self) - nval - ntest
        train = self.conversations[:ntrain]
        dev = self.conversations[ntrain:(ntrain+nval)]
        test = self.conversations[(ntrain+nval):]
        train, dev, test = [Subtask1Dataset(d, self.input_fields, self.target_fields) for d in (train, dev, test)]
        return train, dev, test

