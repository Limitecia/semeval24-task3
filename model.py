
import torch.nn as nn
import torch
from modules import PretrainedEmbedding, LSTM, Biaffine, FFN
from torch.nn.utils.rnn import PackedSequence, pad_sequence
from utils import Config
from typing import Tuple, Optional

class EmotionCausalModel(nn.Module):
    def __init__(
        self,
        pretrained: str,
        word_config: Config,
        spk_config: Config,
        em_config: Config,
        ut_embed_size,
        finetune: bool = False
    ):
        super().__init__()
        self.word_embed = PretrainedEmbedding(pretrained, word_config.embed_size, word_config.pad_index, finetune=finetune)
        self.ut_embed = nn.LSTM(input_size=word_config.embed_size, hidden_size=ut_embed_size,
                                bidirectional=True, dropout=0.1, num_layers=1, batch_first=True)
        self.spk_embed = nn.Embedding(
            num_embeddings=spk_config.vocab_size, embedding_dim=spk_config.embed_size, padding_idx=spk_config.pad_index)
        self.em_embed = nn.Embedding(
            num_embeddings=em_config.vocab_size, embedding_dim=em_config.embed_size, padding_idx=em_config.pad_index
        )
        self.em_pad_index = em_config.pad_index

        self.ut_cause = FFN(in_features=ut_embed_size*2+spk_config.embed_size,
                            out_features=ut_embed_size, activation=nn.LeakyReLU(0.1))
        self.ut_effect = FFN(in_features=ut_embed_size*2+spk_config.embed_size,
                             out_features=ut_embed_size, activation=nn.LeakyReLU(0.1))
        self.em_cause = FFN(in_features=ut_embed_size*2+spk_config.embed_size,
                            out_features=ut_embed_size, activation=nn.LeakyReLU(0.1))
        self.em_effect = FFN(in_features=ut_embed_size*2+spk_config.embed_size,
                             out_features=ut_embed_size, activation=nn.LeakyReLU(0.1))
        self.ut_attn = Biaffine(n_in=ut_embed_size, n_out=1, bias_x=True, bias_y=False)
        self.em_attn = Biaffine(n_in=ut_embed_size, n_out=em_config.vocab_size, bias_x=True, bias_y=True)

        self.span = LSTM(
            input_size=word_config.embed_size + em_config.embed_size,
            hidden_size=word_config.embed_size//2, output_size=1, activation=nn.Sigmoid(),
            bidirectional=True, num_layers=1
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        words: torch.Tensor,
        speakers: torch.Tensor,
        emotions: torch.Tensor,
        graphs: torch.Tensor,
        spans: torch.Tensor,
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
        word_embed, s_ut, s_em = self.encode(words, speakers)

        # Spans prediction
        cause_mask = graphs | (s_ut > 0)
        b, effect, cause = cause_mask.nonzero(as_tuple=True)
        em_embed = self.em_embed(emotions)[b, effect]
        word_cause = word_embed[b, cause]
        span_embed = torch.concat([word_cause, em_embed.unsqueeze(1).repeat(1, word_cause.shape[1], 1)], dim=-1)
        span_preds = self.span(span_embed).squeeze(-1)
        s_span = torch.zeros_like(spans, dtype=torch.float32)
        s_span[b, effect, cause] = span_preds
        return s_ut, s_em, s_span


    def encode(self, words: torch.Tensor, speakers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # compute word embeddings
        batch_size, max_conv_len, max_ut_len = words.shape[0], words.shape[1], words.shape[2]

        # word_embed ~ [batch_size, max(conv_len), max(ut_len), word_embed_size]
        word_embed = torch.stack([self.word_embed(words[i]) for i in range(batch_size)], dim=0)

        # spk_embed ~ [batch_size, max(conv_len), spk_embed_size]
        spk_embed = self.spk_embed(speakers)

        # ut_embed ~ [batch_size, max(conv_len), ut_embed_size]
        ut_embed = torch.stack([torch.concat(self.ut_embed(word_embed[i])[1][0].unbind(0), dim=-1) for i in range(batch_size)], dim=0)
        ut_embed = torch.concat([ut_embed, spk_embed], dim=-1)

        # compute cause-effect representations
        ut_cause = self.ut_cause(ut_embed)          # [batch_size, pad(conv_len), ut_embed_size]
        ut_effect = self.ut_effect(ut_embed)        # [batch_size, pad(conv_len), ut_embed_size]
        em_cause = self.em_cause(ut_embed)          # [batch_size, pad(conv_len), ut_embed_size]
        em_effect = self.em_effect(ut_embed)        # [batch_size, pad(conv_len), ut_embed_size]

        s_ut = self.ut_attn(ut_effect, ut_cause)
        s_em = torch.permute(self.em_attn(em_effect, em_cause), (0, 2, 3,1))
        return word_embed, s_ut, s_em




    def loss(
        self,
        s_ut: torch.Tensor,
        s_em: torch.Tensor,
        s_span: torch.Tensor,
        graphs: torch.Tensor,
        spans: torch.Tensor,
        emotions: torch.Tensor,
        pad_mask: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Args:
            s_em (torch.Tensor): ``[batch_size, max(conv_len), max(conv_len)]``
            s_span (torch.Tensor): ``[n_arcs, max(ut_len)]``
            graphs (torch.Tensor): ``[batch_size, max(conv_len), max(conv_len)]``
            spans (torch.Tensor): ``[batch_size, max(conv_len), max(conv_len), max(ut_len)]``
            pad_mask (torch.Tensor): ``[batch_size, max(conv_len)]``
        """
        # compute utterance unlabeled cause-relation
        ut_loss = self.criterion(s_ut[pad_mask], graphs[pad_mask].to(torch.float32))

        # compute utterance labeled cause-relation
        em_mask = (s_ut > 0) | graphs
        em_mask[torch.ones_like(em_mask, dtype=torch.bool).triu()] = False
        em_mask[~pad_mask] = False
        b, effect, cause = em_mask.nonzero(as_tuple=True)
        ems = torch.zeros_like(graphs).to(torch.int32)
        ems[b, effect, cause] = emotions[b, effect]
        em_loss = self.criterion(s_em[pad_mask].flatten(0,1), ems[pad_mask].flatten().to(torch.long))

        # compute spans loss
        span_loss = self.criterion(s_span[em_mask], spans[em_mask].to(torch.float32))

        return ut_loss + em_loss + span_loss

    def predict(
        self,
        words: torch.Tensor,
        speakers: torch.Tensor,
        pad_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            words (torch.Tensor): ``[batch_size, max(conv_len), max(ut_len), fix_len]``
            speakers (torch.Tensor): ``[batch_size, max(conv_len)]``
            pad_mask (torch.Tensor0: ``[batch_size, max(conv_len)]``
        Returns:
            em_preds (torch.Tensor): ``[batch_size, max(conv_len), max(conv_len)]``
            s_span (torch.Tensor): ``[batch_size, max(conv_len), max(conv_len), max(ut_len)]``
        """
        word_embed, s_ut, s_em = self.encode(words, speakers)
        s_ut[torch.ones_like(s_ut, dtype=torch.bool).triu()] = -1


        ut_preds = s_ut > 0
        ut_preds[pad_mask, :] = False
        em_preds = s_em[:, :, :, 1:].argmax(-1)
        em_preds[~ut_preds] = self.em_pad_index


        # compute spans
        b, effect, cause = ut_preds.nonzero(as_tuple=True)
        word_embed = word_embed[b, cause]
        em_embed = self.em_embed(em_preds[b, effect, cause])
        span_embed = torch.concat([word_embed, em_embed.unsqueeze(1).repeat(1, word_embed.shape[1], 1)], dim=-1)
        span_preds = self.span(span_embed).squeeze(-1)
        s_span = torch.zeros(*iter(ut_preds.shape), span_preds.shape[-1], dtype=torch.bool, device=ut_preds.device)
        s_span[b, effect, cause] = (span_preds > 0.5)

        return em_preds, s_span

