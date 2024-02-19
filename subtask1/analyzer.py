from __future__ import annotations

import os, torch, pickle, shutil
import torch.nn as nn
from typing import List, Tuple, Union, Callable
from data import Subtask1Dataset, Conversation, CauseRelation
from utils import *
from torch.utils.data import DataLoader, Dataset
from subtask1.model import Subtask1Model
from tqdm import tqdm
from utils.metric import Subtask1Metric
from torch.optim import AdamW, Optimizer, Adam, RMSprop
from torch.optim.lr_scheduler import StepLR
import numpy as np 
from analyzer import Analyzer 

class Subtask1Analyzer(Analyzer):
    METRIC = Subtask1Metric 
    INPUT_FIELDS = ['TEXT', 'SPEAKER']
    TARGET_FIELDS = ['EMOTION', 'GRAPH', 'SPAN']
    MODEL = Subtask1Model

    def train_step(self, inputs: Tuple[torch.Tensor], targets: Tuple[torch.Tensor], masks: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emotions, graphs, spans, pad_mask, span_mask = *targets, *masks
        scores = self.model(*inputs, pad_mask, graphs)
        return self.model.loss(*scores, emotions, graphs, spans > 0, pad_mask, span_mask)

    @torch.no_grad()
    def pred_step(self, inputs: Tuple[torch.Tensor], masks: Tuple[torch.Tensor], convs: List[Conversation]) -> List[Conversation]:
        words, speakers, pad_mask, span_mask = *inputs, *masks
        lens = pad_mask.sum(-1).tolist()
        ut_preds, em_preds, span_preds = self.model.predict(words, speakers, pad_mask, span_mask)
        em_preds = em_preds[pad_mask].split(lens)
        ut_preds = [ut_pred[:l, :l] for ut_pred, l in zip(ut_preds.unbind(0), lens)]
        span_preds = [span_pred[:l, :l, :(conv.max_len+1)] for span_pred, l, conv in zip(span_preds.unbind(0), lens, convs)]
        preds = [conv.update1(graph, self.EMOTION.batch_decode(em), span) for conv, graph, em, span in zip(convs, ut_preds, em_preds, span_preds)]
        return preds 

    def transform(self, batch) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[Conversation]]:
        inputs, targets, conv = zip(*batch)
        words, speakers = [t.batch_encode(input) for t, input in zip(self.input_tkzs, zip(*inputs))]
        emotions, graphs, spans = [t.batch_encode(target) for t, target in zip(self.target_tkzs, zip(*targets))]
        pad_mask = (speakers != self.SPEAKER.pad_index)
        span_mask = (spans != self.SPAN.pad_index)
        return (words, speakers), (emotions, graphs, spans), (pad_mask, span_mask), conv
    
    @classmethod
    def build(
        cls,
        data: Subtask1Dataset,
        text_conf: Config,
        device: str, 
        ut_embed_size: int = 400,
        spk_embed_size: int = 50,
    ) -> Subtask1Analyzer:
        input_tkzs = [
            TextTokenizer('TEXT', text_conf.pretrained, lower=False, bos=True, eos=False), 
            PositionalTokenizer('SPEAKER', max(map(len, data.convs)))
        ]
        target_tkzs = [
            Tokenizer('EMOTION', lower=True, max_words=None), 
            GraphTokenizer('GRAPH'), 
            SpanTokenizer('SPAN')
        ]

        for tkz in filter(lambda x: x.TRAINABLE, [*input_tkzs, *target_tkzs]):
            tkz.fit(flatten_list([getattr(conv, tkz.field) for conv in data.convs]))

        # construct model
        text_conf.pad_index = input_tkzs[0].pad_index
        spk_conf = Config(vocab_size=len(input_tkzs[1]), embed_size=spk_embed_size, pad_index=input_tkzs[1].pad_index)
        em_conf = Config(vocab_size=len(target_tkzs[0]), pad_index=target_tkzs[0].pad_index, weights=target_tkzs[0].weights)
        model = Subtask1Model(ut_embed_size, text_conf, spk_conf, em_conf, device)

        analyzer = Subtask1Analyzer(model, input_tkzs, target_tkzs)
        return analyzer



