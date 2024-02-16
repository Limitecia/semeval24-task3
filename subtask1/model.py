
import torch.nn as nn
import torch
from modules import PretrainedEmbedding, Biaffine, FFN, LSTM
from torch.nn.utils.rnn import PackedSequence, pad_sequence
from utils import Config, to, expand_mask, normalize
from typing import Tuple, Optional
import numpy as np 

class Subtask1Model(nn.Module):
    LAYERS = ['ut_attn', 'em_attn', 'span_attn', 'span', 'ut_effect', 'ut_cause', 'em_effect', 'em_cause']
    PARAMS = ['ut_embed_size', 'device', 'text_conf', 'spk_conf', 'em_conf']
    SPAN_THRESHOLD = 0

    def __init__(
        self,
        ut_embed_size,
        device: str,
        text_conf: Config,
        spk_conf: Config,
        em_conf: Config
    ):
        super().__init__()
        self.word_embed = PretrainedEmbedding(**text_conf())
        text_conf.embed_size = self.word_embed.embed_size
        self.spk_embed = nn.Embedding(spk_conf.vocab_size, spk_conf.embed_size, spk_conf.pad_index).to(text_conf.device)

        self.ut_cause = FFN(text_conf.embed_size + spk_conf.embed_size, ut_embed_size)
        self.ut_effect = FFN(text_conf.embed_size + spk_conf.embed_size, ut_embed_size)
        self.ut_attn = Biaffine(n_in=ut_embed_size, n_out=2, bias_x=True, bias_y=True, dropout=0.3)

        self.em_cause = FFN(text_conf.embed_size + spk_conf.embed_size, ut_embed_size)
        self.em_effect = FFN(text_conf.embed_size + spk_conf.embed_size, ut_embed_size)
        self.em_attn = Biaffine(n_in=ut_embed_size, n_out=em_conf.vocab_size, dropout=0.3, bias_x=True, bias_y=True)
        
        # self.span_attn = nn.MultiheadAttention(
        #     text_conf.embed_size, num_heads=num_heads, dropout=0.1, batch_first=True, 
        #     kdim=em_config.embed_size, vdim=em_config.embed_size)
        self.span_attn = nn.MultiheadAttention(
            text_conf.embed_size, num_heads=1, dropout=0.1, batch_first=True, 
            kdim=ut_embed_size, vdim=ut_embed_size
        )
        self.span = FFN(text_conf.embed_size, 1)
        nn.init.orthogonal_(self.span_attn.out_proj.weight)
        nn.init.zeros_(self.span_attn.out_proj.bias)

        self.criterion = nn.CrossEntropyLoss()
        self.span_criterion = nn.BCEWithLogitsLoss()
        self.device = device

        for layer in self.LAYERS:
            self.__setattr__(layer, self.__getattr__(layer).to(device))
        for name, value in locals().items():
            if name in self.PARAMS:
                self.__setattr__(name, value)


    def forward(
        self,
        words: torch.Tensor,
        speakers: torch.Tensor,
        pad_mask: torch.Tensor,
        graphs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Implements training step.
        Args:
            words (torch.Tensor): ``[batch_size, max(conv_len), max(ut_len), fix_len]``
            speakers (torch.Tensor): ``[batch_size, max(conv_len)]``
            emotions (torch.Tensor): ``[batch_size, max(conv_len)]``
            graphs (torch.Tensor): ``[batch_size, max(conv_len), max(conv_len)]``
            spans (torch.Tensor): ``[batch_size,, max(conv_len), max(conv_len), max(ut_len)]``

        Returns:
            ~torch.Tensor, ~torch.Tensor, ~torch.Tensor
            - s_ut: ``[batch_size, max(conv_len), max(conv_len)]``
            - s_em: ``[batch_size, max(conv_len), max(conv_len), n_emotions]``
            - s_span: ``[batch_size,, max(conv_len), max(conv_len), max(ut_len)]``
        """
        word_embed, ut_embed = self.encode(words, speakers)
        s_ut, s_em, s_span = self.decode(word_embed, ut_embed, pad_mask, graphs)
        return s_ut, s_em, s_span


    def encode(
            self, 
            words: torch.Tensor, 
            speakers: torch.Tensor, 
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        # compute word embeddings
        words, speakers = to(self.text_conf.device, words, speakers)
        batch_size, *_ = words.shape[0], words.shape[1], words.shape[2]

        # word_embed ~ [batch_size, max(conv_len), max(ut_len), word_embed_size,]
        word_embed = torch.stack([self.word_embed(words[i]) for i in range(batch_size)], dim=0)

        # spk_embed ~ [batch_size, max(conv_len), spk_embed_size]
        spk_embed = self.spk_embed(speakers)

        # ut_embed ~ [batch_size, max(conv_len), ut_embed_size]
        ut_embed = torch.concat([word_embed[:, :, 0], spk_embed], dim=-1)
        return to(self.device, word_embed[:, :, 1:], ut_embed)

    def decode(
            self, 
            word_embed: torch.Tensor, 
            ut_embed: torch.Tensor,
            pad_mask: torch.Tensor,
            graphs: Optional[torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # compute cause-effect representations
        ut_cause = self.ut_cause(ut_embed)
        ut_effect = self.ut_effect(ut_embed)
        s_ut = self.ut_attn(ut_cause, ut_effect).permute(0, 2, 3, 1)

        # compute emotion predictions using utterance representations
        em_cause = self.em_cause(ut_embed)
        em_effect = self.em_effect(ut_embed)
        s_em = self.em_attn(em_cause, em_effect).permute(0, 2, 3, 1)

        # compute spans 
        ut_mask = s_ut.argmax(-1).to(torch.bool) if graphs is None else graphs.to(self.device)
        b, cause, effect = (ut_mask & expand_mask(pad_mask).to(self.device)).nonzero(as_tuple=True)
        scores, _ = self.span_attn(word_embed[b, cause], em_effect[b, effect].unsqueeze(1), em_effect[b, cause].unsqueeze(1))
        s_span = torch.zeros(word_embed.shape[0], word_embed.shape[1], word_embed.shape[1], word_embed.shape[2], device=self.device) - 1
        s_span[b, cause, effect] = self.span(scores).squeeze(-1)
        return s_ut, s_em, s_span



    def loss(
        self,
        s_ut: torch.Tensor,
        s_em: torch.Tensor,
        s_span: torch.Tensor,
        emotions: torch.Tensor, 
        graphs: torch.Tensor,
        spans: torch.Tensor,
        pad_mask: torch.Tensor,
        span_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        graphs, spans, emotions = to(self.device, graphs, spans, emotions)

        # compute utterance unlabeled cause-relation
        ut_mask = expand_mask(pad_mask).to(self.device)
        # ut_mask = expand_mask(pad_mask).to(self.device)
        ut_loss = self.criterion(s_ut[ut_mask], graphs[ut_mask].to(torch.long))

        # compute utterance labeled cause-relation gold
        ems = torch.zeros_like(graphs, dtype=torch.int32, device=graphs.device)
        b, cause, effect = graphs.nonzero(as_tuple=True)
        ems[b, cause, effect] = emotions[b, effect]
        em_loss = self.criterion(s_em[ut_mask], ems[ut_mask].to(torch.long))
        # em_loss = self.em_criterion(s_em[pad_mask], emotions[pad_mask].to(torch.long))

        # compute spans loss
        span_mask[~ut_mask] = False
        span_loss = self.span_criterion(s_span[span_mask].flatten(), spans[span_mask].flatten().to(torch.float32))
        # f = 10**int(np.log10(ut_loss.item()))
        # return ut_loss, em_loss*f*int(ut_loss/f), span_loss
        return ut_loss, em_loss, span_loss

    def predict(
        self,
        words: torch.Tensor,
        speakers: torch.Tensor,
        pad_mask: torch.Tensor,
        span_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_ut, s_em, s_span = self(words, speakers, pad_mask)
        ut_preds, em_preds = s_ut.argmax(-1).to(torch.bool), s_em.mean(1)[:, :, 2:].argmax(-1) + 2
        ut_preds[~expand_mask(pad_mask).to(self.device)] = False
        s_span[~span_mask] = 0
        span_preds = torch.zeros_like(s_span, dtype=torch.bool)
        if ut_preds.sum() > 0:
            span_preds[ut_preds] = torch.tensor(np.apply_along_axis(smooth, -1, s_span[ut_preds].cpu().numpy()), device=s_span.device)
        return ut_preds, em_preds, span_preds



def smooth(x: np.ndarray):
    mask = np.zeros_like(x, dtype=np.bool_)
    if (x > Subtask1Model.SPAN_THRESHOLD).sum() < 1:
        arg1 = np.argsort(x)[0]
        mask[(arg1-1):(arg1+1)] = True
    else:
        nonzero = (x > Subtask1Model.SPAN_THRESHOLD).flatten().nonzero()[0].tolist()
        start, end = nonzero[0], nonzero[-1]
        mask[start:end] = True
    return mask
        
    
    
