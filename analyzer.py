from __future__ import annotations

import os, torch
import torch.nn as nn
from typing import List, Tuple
from data import Subtask1Dataset
from utils import Tokenizer, PretrainedTokenizer, NullTokenizer, Config, parallel, TensorTokenizer
from torch.utils.data import DataLoader, Dataset
from model import EmotionCausalModel
from torch.nn.utils.rnn import pad_sequence

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

    def train(
        self,
        train: Subtask1Dataset,
        dev: Subtask1Dataset,
        test: Subtask1Dataset,
        epochs: int = 100,
        batch_size: int = 100
    ):
        train_dl, dev_dl, test_dl = [
            DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=self.transform)
            for data in (train, dev, test)
        ]

        for epoch in range(epochs):
            for i, (inputs, targets) in enumerate(train_dl):
                self.train_step(inputs, targets)


    def train_step(self, inputs: List[torch.Tensor], targets: List[torch.Tensor]) -> torch.Tensor:
        words, speakers, emotions, graphs, spans = zip(*[*inputs, *targets])



    @torch.no_grad()
    def eval_step(self, inputs: List[torch.Tensor], targets: List[torch.Tensor]):
        pass



    def transform(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor]]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        inputs, targets = zip(*batch)
        inputs = [t.encode_batch(input) for t, input in zip(self.input_tkzs, zip(*inputs))]
        targets = [t.encode_batch(target) for t, target in zip(self.target_tkzs, zip(*targets))]
        return inputs, targets


    @classmethod
    def build(
        cls,
        data: str,
        pval: float,
        ptest: float,
        pretrained: str,
        num_workers: int = os.cpu_count(),
        show: bool = True,
        device: str = 'cuda:0',
        word_embed_size: int = 200,
        spk_embed_size: int = 20,
        em_embed_size: int = 20,
        pos_embed_size: int = 20,
        ut_embed_size: int = 200
    ) -> Tuple[EmotionCausaAnalyzer, Tuple[Subtask1Dataset, Subtask1Dataset, Subtask1Dataset]]:
        # create tokenizers
        input_tkzs = [
            PretrainedTokenizer('words', 'bert-base-uncased', lower=False, fix_len=5),
            Tokenizer('SPEAKER', lower=True, max_words=None)
        ]

        target_tkzs = [
            Tokenizer('EMOTION', lower=True, max_words=None),
            TensorTokenizer('GRAPH'),
            NullTokenizer('SPAN')
        ]

        # read dataset and split
        data = Subtask1Dataset.from_path(
            data, [t.field for t in input_tkzs], [t.field for t in target_tkzs], build=False)
        train, dev, test = data.split(pval, ptest)

        # train tokenizers only with train data
        tkzs = parallel([*input_tkzs, *target_tkzs], train.conversations, num_workers, show)
        input_tkzs = [tkzs.pop(0) for _ in range(len(input_tkzs))]
        target_tkzs = tkzs

        # construct model
        args = Config(
            pos_embed_size=pos_embed_size, max_len=data.max_len, avg_span_len=int(train.avg_span_len),
            pretrained=pretrained, ut_embed_size=ut_embed_size
        )
        args.word_config = Config(embed_size=word_embed_size, pad_index=input_tkzs[0].pad_index)
        args.spk_config = Config(vocab_size=len(input_tkzs[1]), embed_size=spk_embed_size, pad_index=input_tkzs[1].pad_index)
        args.em_config = Config(vocab_size=len(target_tkzs[0]), embed_size=em_embed_size, pad_index=target_tkzs[0].pad_index)
        model = EmotionCausalModel(**args()).to(device)

        analyzer = EmotionCausaAnalyzer(model, input_tkzs, target_tkzs, device)
        return analyzer, (train, dev, test)




if __name__ == '__main__':
    analyzer, (train, dev, test) = EmotionCausaAnalyzer.build(
        data='dataset/text/Subtask_1_train.json', pval=0.2, ptest=0.1, pretrained='bert-base-uncased'
    )
    analyzer.train(train, dev, test)



