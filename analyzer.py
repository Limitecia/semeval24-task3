from __future__ import annotations

import os, torch
import torch.nn as nn
from typing import List, Tuple, Union, Callable
from data import Subtask1Dataset, Conversation
from utils import Tokenizer, WordTokenizer, GraphTokenizer, SpanTokenizer, Config, parallel, to, flatten_list
from torch.utils.data import DataLoader, Dataset
from model import EmotionCausalModel
from tqdm import tqdm
from metric import Subtask1Metric
from torch.optim import Adam, Optimizer


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
        self.optimizer = None

    def train(
        self,
        train: Subtask1Dataset,
        dev: Subtask1Dataset,
        test: Subtask1Dataset,
        optimizer: Callable = Adam,
        lr: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 100,
        show: bool = True
    ):
        train_dl, dev_dl, test_dl = [
            DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=self.transform)
            for data in (train, dev, test)
        ]
        self.optimizer = optimizer(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            train_loss = self.forward(epoch, train_dl, optimize=True)
            dev_metric = self.forward(epoch, dev_dl, optimize=False)
            test_metric = self.forward(epoch, test_dl, optimize=False)
            if show:
                print(f'\nEpoch {epoch} [train]: loss={round(train_loss, 2)}')
                print(f'Epoch {epoch} [dev]: {repr(dev_metric)}')
                print(f'Epoch {epoch} [test]: {repr(test_metric)}')
            del train_loss, dev_metric, test_metric


    def forward(self, epoch: int, loader: DataLoader, show: bool = True, optimize: bool = True) -> Union[float, Subtask1Metric]:
        global_loss = 0.0
        metric = Subtask1Metric()
        self.model.zero_grad()
        with tqdm(total=len(loader), disable=not show) as bar:
            bar.set_description(f'Epoch {epoch} ({"train" if optimize else "eval"})')
            for i, (inputs, targets) in enumerate(loader):
                if optimize:
                    loss = self.train_step(inputs, targets)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    loss = loss.item()
                    global_loss += loss
                else:
                    metric += self.eval_step(inputs, targets)
                    loss = metric.loss
                bar.update(1)
                bar.set_postfix({'loss': round(loss, 2)})
                del loss, inputs, targets
                torch.cuda.empty_cache()
            bar.close()
        return global_loss / len(loader) if optimize else metric

    def train_step(self, inputs: List[torch.Tensor], targets: List[torch.Tensor]) -> torch.Tensor:
        r"""
        Implements training step.
        Args:
            inputs (List[torch.Tensor]):
                - words (torch.Tensor): ``[batch_size, max(conv_len), max(ut_len), fix_len]``
                - speakers (torch.Tensor): ``[batch_size, max(conv_len)]``
                - emotions (torch.Tensor): ``[batch_size, max(conv_len)]``
            targets (List[torch.Tensor]): `
                - graphs (torch.Tensor): ``[batch_size, max(conv_len), max(conv_len)]``
                - spans (torch.Tensor): ``[batch_size, max(conv_len), max(conv_len), max(ut_len)]``
        Returns:
            ~torch.Tensor: Loss.
        """
        words, speakers, emotions, graphs, spans = to([*inputs, *targets], self.device)
        pad_mask = (speakers != self.input_tkzs[1].pad_index)

        s_ut, s_em, s_span = self.model(words, speakers, emotions, graphs, spans)
        loss = self.model.loss(s_ut, s_em, s_span, graphs, spans, emotions, pad_mask)
        return loss

    @torch.no_grad()
    def eval_step(self, inputs: List[torch.Tensor], targets: List[torch.Tensor]):
        words, speakers, emotions, graphs, spans = to([*inputs, *targets], self.device)
        pad_mask = (speakers != self.input_tkzs[1].pad_index)
        s_ut, s_em, s_span = self.model(words, speakers, emotions, graphs, spans)
        loss = self.model.loss(s_ut, s_em, s_span, graphs, spans, emotions, pad_mask)

        em_preds, span_preds = self.model.predict(words, speakers, pad_mask)

        b, effect, cause = graphs.nonzero(as_tuple=True)
        ems = torch.zeros_like(graphs, dtype=torch.int32)
        ems[graphs] = emotions[b, cause]
        # metric = Subtask1Metric(loss, em_preds, span_preds, ems, spans, pad_mask, self.model.em_pad_index)
        metric = Subtask1Metric(loss, em_preds, span_preds, ems, spans, pad_mask, self.model.em_pad_index)
        del words, speakers, emotions, graphs, spans, pad_mask, s_ut, s_em, em_preds, span_preds, ems
        return metric


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
        device: str = 'cuda:0',
        word_embed_size: int = 100,
        spk_embed_size: int = 10,
        em_embed_size: int = 10,
        ut_embed_size: int = 100,
        **kwargs
    ) -> Tuple[EmotionCausaAnalyzer, Tuple[Subtask1Dataset, Subtask1Dataset, Subtask1Dataset]]:
        # create tokenizers
        input_tkzs = [
            WordTokenizer('TEXT', pretrained, lower=False, bos=True),
            Tokenizer('SPEAKER', lower=True, max_words=None),
            Tokenizer('EMOTION', lower=True, max_words=None)
        ]

        target_tkzs = [
            # GraphTokenizer('GRAPH', vocab=set(), pad_token=Conversation.NO_CAUSE),
            GraphTokenizer('GRAPH'),
            SpanTokenizer('SPAN')
        ]

        # read dataset and split
        data = Subtask1Dataset.from_path(
            data, [t.field for t in input_tkzs], [t.field for t in target_tkzs], build=False)
        train, dev, test = data.split(pval, ptest)

        # train tokenizers only with train data
        for tkz in [*input_tkzs, *target_tkzs]:
            if tkz.TRAINABLE:
                tkz.fit(flatten_list([getattr(conv, tkz.field) for conv in train.conversations]))

        # target_tkzs[0] = GraphTokenizer('GRAPH', vocab=input_tkzs[-1].tokens, pad_token=Conversation.NO_CAUSE)

        # construct model
        args = Config(
            pretrained=pretrained, ut_embed_size=ut_embed_size, cls_index=input_tkzs[0].bos_index, **kwargs
        )
        args.word_config = Config(embed_size=word_embed_size, pad_index=input_tkzs[0].pad_index)
        args.spk_config = Config(vocab_size=len(input_tkzs[1]), embed_size=spk_embed_size, pad_index=input_tkzs[1].pad_index)
        args.em_config = Config(vocab_size=len(input_tkzs[2]), embed_size=em_embed_size, pad_index=input_tkzs[2].pad_index)
        model = EmotionCausalModel(**args()).to(device)

        analyzer = EmotionCausaAnalyzer(model, input_tkzs, target_tkzs, device)
        return analyzer, (train, dev, test)




if __name__ == '__main__':
    analyzer, (train, dev, test) = EmotionCausaAnalyzer.build(
        data='dataset/text/Subtask_1_train.json', pval=0.2, ptest=0.1, pretrained='roberta-large', finetune=True
    )
    analyzer.train(train, dev, test, batch_size=3, lr=5e-3)



