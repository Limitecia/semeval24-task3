from __future__ import annotations
import torch 
import torch.nn as nn
from typing import List, Tuple
from data import Subtask2Dataset, Conversation
from utils import *
from subtask2.model import Subtask2Model
from utils.metric import Subtask2Metric
from torch.optim.lr_scheduler import StepLR
from analyzer import Analyzer


class Subtask2Analyzer(Analyzer):
    METRIC = Subtask2Metric
    MODEL = Subtask2Model 
    INPUT_FIELDS = ['TEXT', 'SPEAKER', 'FRAME', 'AUDIO']
    TARGET_FIELDS = ['EMOTION', 'GRAPH']

    def train_step(
            self, 
            inputs: Tuple[torch.Tensor], 
            targets: Tuple[torch.Tensor], 
            masks: Tuple[torch.Tensor]
        ) -> torch.Tensor:
        scores = self.model(*inputs)
        return self.model.loss(*scores, *targets, *masks)

    @torch.no_grad()
    def pred_step(
            self, 
            inputs: Tuple[torch.Tensor], 
            masks: Tuple[torch.Tensor], 
            convs: List[Conversation]
        ) -> List[Conversation]:
        pad_mask, *_ = masks
        lens = pad_mask.sum(-1).tolist()
        ut_preds, em_preds = self.model.predict(*inputs, *masks)
        em_preds = em_preds[pad_mask].split(lens)
        ut_preds = [ut_pred[:l, :l] for ut_pred, l in zip(ut_preds.unbind(0), lens)]
        preds = [conv.update2(graph, self.EMOTION.batch_decode(em)) for conv, graph, em in zip(convs, ut_preds, em_preds)]
        return preds 

    def transform(self, batch) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[Conversation]]:
        inputs, targets, conv = zip(*batch)
        words, speakers, frames, audios = [t.batch_encode(input) for t, input in zip(self.input_tkzs, zip(*inputs))]
        targets = [t.batch_encode(target) for t, target in zip(self.target_tkzs, zip(*targets))]
        pad_mask = (speakers != self.SPEAKER.pad_index)
        return (words, speakers, frames, audios), targets, (pad_mask,), conv


    @classmethod
    def build(
        cls,
        data: Subtask2Dataset,
        text_conf: Config, 
        img_conf: Optional[Config],
        audio_conf: Optional[Config], 
        device: str,
        ut_embed_size: int = 400,
        spk_embed_size: int = 50
    ) -> Subtask2Analyzer:
        # create tokenizers
        input_tkzs = [
            TextTokenizer('TEXT', text_conf.pretrained, lower=False, bos=True, eos=False),
            PositionalTokenizer('SPEAKER', max(map(len, data.convs))),
            ImageProcessor('FRAME', img_conf.pretrained, num_frames=img_conf.pop('num_frames')) if img_conf is not None else RawTokenizer('FRAME'),
            AudioProcessor('AUDIO', audio_conf.pretrained) if audio_conf is not None else RawTokenizer('AUDIO')
        ]
        target_tkzs = [Tokenizer('EMOTION', lower=True, max_words=None), GraphTokenizer('GRAPH')]

        # train tokenizers only with train data
        for tkz in filter(lambda x: x.TRAINABLE, [*input_tkzs, *target_tkzs]):
            tkz.fit(flatten_list([getattr(conv, tkz.field) for conv in data.convs]))
            
        # construct model
        text_conf.pad_index = input_tkzs[0].pad_index
        spk_conf = Config(vocab_size=len(input_tkzs[1]), embed_size=spk_embed_size, pad_index=input_tkzs[1].pad_index)
        em_conf = Config(vocab_size=len(target_tkzs[0]), pad_index=target_tkzs[0].pad_index, weights=target_tkzs[0].weights)
        model = Subtask2Model(ut_embed_size, device, text_conf, spk_conf, img_conf, audio_conf, em_conf)

        analyzer = Subtask2Analyzer(model, input_tkzs, target_tkzs)
        return analyzer

