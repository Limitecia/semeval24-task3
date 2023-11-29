from __future__ import annotations

import os

import torch.nn as nn
from typing import List, Union
from data import Subtask1Dataset
from utils import Tokenizer, PretrainedTokenizer, NullTokenizer, StaticTokenizer
from torch.utils.data import DataLoader, Dataset


class EmotionCausaAnalyzer:
    def __init__(
        self,
        model: nn.Module,
        input_tkzs: List[Tokenizer],
        target_tkzs: List[Tokenizer],
        device: str
    ):
        self.model = model
        self.input_tkzs = input_tkzs
        self.target_tkzs = target_tkzs
        self.device = device


    @classmethod
    def build(
        cls,
        data: str,
        pval: float,
        ptest: float,
        num_workers: int = os.cpu_count(),
        show: bool = True
    ):
        # create tokenizers
        input_tkzs = [
            PretrainedTokenizer('TEXT', 'bert-base-uncased', lower=False, fix_len=5),
            Tokenizer('SPEAKER', lower=True, max_words=None)
        ]

        target_tkzs = [
            Tokenizer('EMOTION', lower=True, max_words=None),
            NullTokenizer('GRAPH'),
            NullTokenizer('SPAN')
        ]

        # read dataset and split
        data = Subtask1Dataset.from_path(data, input_tkzs, target_tkzs, build=False)
        train, dev, test = data.split(pval, ptest)

        # train tokenizers only with train data
        input_tkzs, target_tkzs = train.build(num_workers, show)
        target_tkzs.append(StaticTokenizer('EMOTIONS', vocab=list(target_tkzs[0].tokens)))
        dev.load(input_tkzs, target_tkzs)
        test.load(input_tkzs, target_tkzs)



