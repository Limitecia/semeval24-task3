from __future__ import annotations

import os, torch
import torch.nn as nn
from typing import List, Tuple, Union, Callable
from data import Subtask1Dataset, Conversation, CauseRelation
from utils import Tokenizer, WordTokenizer, GraphTokenizer, SpanTokenizer, Config, flatten_list, to
from torch.utils.data import DataLoader, Dataset
from model import EmotionCausalModel
from tqdm import tqdm
from metric import Subtask1Metric
from torch.optim import AdamW, Optimizer


class EmotionCausaAnalyzer:
    def __init__(
        self,
        model: nn.Module,
        input_tkzs: List[Tokenizer],
        target_tkzs: List[Tokenizer]
    ):
        self.model = model
        self.input_tkzs = input_tkzs
        self.target_tkzs = target_tkzs
        self.optimizer = None

    def train(
        self,
        train: Subtask1Dataset,
        dev: Subtask1Dataset,
        optimizer: Callable = AdamW,
        lr: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 100,
        batch_update: int = 1, 
        show: bool = True
    ):
        train_dl, dev_dl = [
            DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=self.transform)
            for data in (train, dev)
        ]
        self.optimizer = optimizer(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            train_loss = self.forward(epoch, train_dl, batch_update, optimize=True)
            train_metric = self.forward(epoch, train_dl, batch_update, optimize=False)
            dev_metric = self.forward(epoch, dev_dl, batch_update, optimize=False)
            if show:
                print(f'Epoch {epoch} [train]: loss={round(float(train_loss), 5)}')
                print(f'Epoch {epoch} [train]: {repr(train_metric)}')
                print(f'Epoch {epoch} [dev]: {repr(dev_metric)}')
            del train_loss, dev_metric


    def forward(self, epoch: int, loader: DataLoader, batch_update: int = 1, show: bool = True, optimize: bool = True) -> Union[float, Subtask1Metric]:
        global_loss = 0.0
        loss = 0.0
        metric = Subtask1Metric()
        self.model.zero_grad()
        with tqdm(total=len(loader), disable=not show) as bar:
            bar.set_description(f'Epoch {epoch} ({"train" if optimize else "eval"})')
            for i, (inputs, targets, pairs) in enumerate(loader):
                if optimize:
                    batch_loss = self.train_step(inputs, targets, pairs)
                    loss += batch_loss
                    if i % batch_update == 0:
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        global_loss += loss
                        loss = 0.0
                else:
                    metric += self.eval_step(inputs, targets, pairs)
                    batch_loss = metric.loss
                bar.update(1)
                bar.set_postfix({'loss': round(float(batch_loss), 2)})
                torch.cuda.empty_cache()
            bar.close()
        return global_loss / len(loader) if optimize else metric

    def train_step(self, inputs: List[torch.Tensor], targets: List[torch.Tensor], pairs: List[List[CauseRelation]]) -> torch.Tensor:
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
        words, speakers, emotions, graphs, spans = *inputs, *targets
        pad_mask = (speakers != self.input_tkzs[1].pad_index)

        s_ut, s_em, s_span = self.model(words, speakers, graphs, spans)
        loss = self.model.loss(s_ut, s_em, s_span, emotions, graphs, spans, pad_mask)
        return loss

    @torch.no_grad()
    def eval_step(self, inputs: List[torch.Tensor], targets: List[torch.Tensor], pairs: List[List[CauseRelation]]):
        words, speakers, emotions, graphs, spans = *inputs, *targets
        pad_mask = (speakers != self.input_tkzs[1].pad_index)
        s_ut, s_em, s_span = self.model(words, speakers, graphs, spans)
        loss = self.model.loss(s_ut, s_em, s_span, emotions, graphs, spans, pad_mask)

        ut_preds, em_preds, span_preds = self.model.predict(words, speakers, pad_mask)

        graphs, ut_preds = graphs.to(torch.int32), ut_preds.to(torch.int32)
        b, cause, effect = graphs.nonzero(as_tuple=True)
        graphs[b, cause, effect] = emotions[b, effect]

        b, cause, effect = ut_preds.nonzero(as_tuple=True)
        ut_preds[b, cause, effect] = em_preds.to(torch.int32)[b, effect]

        metric = Subtask1Metric(loss, *to([ut_preds, span_preds, graphs, spans, pad_mask], self.model.device), self.model.em_pad_index)
        return metric


    def transform(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor]]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[CauseRelation]]]:
        inputs, targets, pairs = zip(*batch)
        inputs = [t.encode_batch(input) for t, input in zip(self.input_tkzs, zip(*inputs))]
        targets = [t.encode_batch(target) for t, target in zip(self.target_tkzs, zip(*targets))]
        return inputs, targets, pairs


    @classmethod
    def build(
        cls,
        data: str,
        pval: float,
        pretrained: str,
        word_embed_size: int = 100,
        spk_embed_size: int = 20,
        em_embed_size: int = 100,
        ut_embed_size: int = 100,
        **kwargs
    ) -> Tuple[EmotionCausaAnalyzer, Tuple[Subtask1Dataset, Subtask1Dataset]]:
        # create tokenizers
        input_tkzs = [
            WordTokenizer('TEXT', pretrained, lower=False, bos=True),
            Tokenizer('SPEAKER', lower=True, max_words=None),
            Tokenizer('EMOTION', lower=True, max_words=None)
        ]

        target_tkzs = [
            GraphTokenizer('GRAPH'),
            SpanTokenizer('SPAN')
        ]

        # read dataset and split
        data = Subtask1Dataset.from_path(
            data, [t.field for t in input_tkzs], [t.field for t in target_tkzs])
        train, dev, _ = data.split(pval, 0)

        # train tokenizers only with train data
        for tkz in [*input_tkzs, *target_tkzs]:
            if tkz.TRAINABLE:
                tkz.fit(flatten_list([getattr(conv, tkz.field) for conv in train.conversations]))

        # target_tkzs[0] = GraphTokenizer('GRAPH', vocab=input_tkzs[-1].tokens, pad_token=Conversation.NO_CAUSE)

        # construct model
        args = Config(
            pretrained=pretrained, ut_embed_size=ut_embed_size, **kwargs
        )
        args.word_config = Config(embed_size=word_embed_size, pad_index=input_tkzs[0].pad_index)
        args.spk_config = Config(vocab_size=len(input_tkzs[1]), embed_size=spk_embed_size, pad_index=input_tkzs[1].pad_index)
        args.em_config = Config(vocab_size=len(input_tkzs[2]), embed_size=em_embed_size, pad_index=input_tkzs[2].pad_index)
        model = EmotionCausalModel(**args())

        analyzer = EmotionCausaAnalyzer(model, input_tkzs, target_tkzs)
        print(model)
        return analyzer, (train, dev)




if __name__ == '__main__':
    analyzer, (train, dev) = EmotionCausaAnalyzer.build(
        data='dataset/text/Subtask_1_train.json', pval=0.2, pretrained='roberta-large', finetune=True,
        word_embed_size=200, ut_embed_size=200, device='cuda:0', embed_device='cuda:1'
    )
    analyzer.train(train, dev, batch_size=10, batch_update=1, lr=1e-4, epochs=1000)



