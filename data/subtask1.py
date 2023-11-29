from __future__ import annotations
from typing import List, Tuple
from data.conversation import Conversation
from torch.utils.data import Dataset
import json, torch, os
from utils import Tokenizer, parallel, PretrainedTokenizer, NullTokenizer


class Subtask1Dataset(Dataset):

    def __init__(
        self,
        conversations: List[Conversation],
        input_tkzs: List[Tokenizer],
        target_tkzs: List[Tokenizer]
    ):
        super().__init__()
        self.conversations = conversations
        self.input_tkzs = input_tkzs
        self.target_tkzs = target_tkzs

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        conver = self.conversations[index]
        inputs = [t.encode(getattr(conver, t.field)) for t in self.input_tkzs]
        targets = [t.encode(getattr(conver, t.field)) for t in self.target_tkzs]
        return inputs, targets

    @classmethod
    def from_path(
        cls,
        path: str,
        input_tkzs: List[Tokenizer],
        target_tkzs: List[Tokenizer],
        build: bool,
        num_workers: int = os.cpu_count(),
        show: bool = True
    ) -> Subtask1Dataset:
        with open(path, 'r') as reader:
            data = json.load(reader)

        conversations = [Conversation.from_dict(d) for d in data]
        data = Subtask1Dataset(conversations, input_tkzs, target_tkzs)
        if build:
            data.build(num_workers, show)
        return data

    def build(self, num_workers: int = os.cpu_count(), show: bool = True) -> Tuple[List[Tokenizer], List[Tokenizer]]:
        tokenizers = parallel([*self.input_tkzs, *self.target_tkzs], self.conversations, num_workers=num_workers, show=show)
        self.input_tkzs = [tokenizers.pop(0) for _ in range(len(self.input_tkzs))]
        self.target_tkzs = tokenizers
        return self.input_tkzs, self.target_tkzs

    def load(self, input_tkzs: List[Tokenizer], target_tkzs: List[Tokenizer]):
        self.input_tkzs = input_tkzs
        self.target_tkzs = target_tkzs


    def split(self, pval: float, ptest: float) -> Tuple[Subtask1Dataset, Subtask1Dataset, Subtask1Dataset]:
        nval, ntest = int(len(self)*pval), int(len(self)*ptest)
        ntrain = len(self) - nval - ntest
        train = self.conversations[:ntrain]
        dev = self.conversations[ntrain:(ntrain+nval)]
        test = self.conversations[(ntrain+nval):]
        train, dev, test = [Subtask1Dataset(d, self.input_tkzs, self.target_tkzs) for d in (train, dev, test)]
        return train, dev, test


if __name__ == '__main__':
    path = 'dataset/text/Subtask_1_train.json'

    # word tokenizer

    dataset = Subtask1Dataset.from_path(path, input_tkzs, target_tkzs, build=True)
