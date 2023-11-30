
import torch.nn as nn
import torch
from modules import PretrainedEmbedding, LSTM, Biaffine, FFN
from torch.nn.utils.rnn import PackedSequence, pad_sequence
from utils import Config
from typing import Tuple

class EmotionCausalModel(nn.Module):
    def __init__(
        self,
        pretrained: str,
        word_config: Config,
        spk_config: Config,
        em_config: Config,
        ut_embed_size,
        pos_embed_size: int,
        avg_span_len: int,
        max_len: int,
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
        graphs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Implements training step.
        Args:
            words (torch.Tensor): ``[batch_size, max(conv_len), max(ut_len), fix_len]``
            speakers (torch.Tensor): ``[batch_size, max(conv_len)]``
            emotions (torch.Tensor): ``[batch_size, max(conv_len)]``
            graphs (torch.Tensor): ``[batch_size, max(conv_len), max(conv_len)]``
        Returns:
            ~torch.Tensor, ~torch.Tensor, ~torch.Tensor
            - s_ut: ``[batch_size, max(conv_len), max(conv_len)]``
            - s_em: ``[batch_size, max(conv_len), max(conv_len), n_emotions]``
            - s_span: ``[n_arcs,, max(ut_len)]``
        """
        # compute word embeddings
        batch_size, max_conv_len, max_ut_len = words.shape[0], words.shape[1], words.shape[2]
        ut_mask = ((words != self.word_embed.pad_index).flatten(-2,-1).sum(-1) != 0)
        conv_lens = ut_mask.sum(-1).tolist()

        # word_embed ~ [batch_size, max(conv_len), max(ut_len), word_embed_size]
        word_embed = torch.stack([self.word_embed(words[i]) for i in range(batch_size)], dim=0)

        # spk_embed ~ [batch_size, max(conv_len), spk_embed_size]
        spk_embed = self.spk_embed(speakers)

        # ut_embed ~ [batch_size, max(conv_len), ut_embed_size]
        ut_embed = torch.stack([torch.concat(self.ut_embed(word_embed[i])[1][0].unbind(0), dim=-1) for i in range(batch_size)], dim=0)  # [n_uts, max_ut_len, ut_embed_size]
        ut_embed = torch.concat([ut_embed, spk_embed], dim=-1)

        # compute cause-effect representations
        ut_cause = self.ut_cause(ut_embed)          # [batch_size, pad(conv_len), ut_embed_size]
        ut_effect = self.ut_effect(ut_embed)        # [batch_size, pad(conv_len), ut_embed_size]
        em_cause = self.em_cause(ut_embed)          # [batch_size, pad(conv_len), ut_embed_size]
        em_effect = self.em_effect(ut_embed)        # [batch_size, pad(conv_len), ut_embed_size]


        s_ut = self.ut_attn(ut_cause, ut_effect)
        s_em = self.em_attn(em_cause, em_effect)

        # Spans prediction
        b, effect, cause = graphs.nonzero(as_tuple=True)
        em_embed = self.em_embed(emotions)[b, effect]
        word_cause = word_embed[b, cause]
        span_embed = torch.concat([word_cause, em_embed.unsqueeze(1).repeat(1, word_cause.shape[1], 1)], dim=-1)
        s_span = self.span(span_embed).squeeze(-1)
        return s_ut, s_em, s_span



    def loss(
        self,
        s_ut: torch.Tensor,
        s_em: torch.Tensor,
        s_span: torch.Tensor,
        graphs: torch.Tensor,
        emotions: torch.Tensor,
        spans: torch.Tensor,
        ut_mask: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Args:
            s_ut (torch.Tensor): ``[batch_size, max(conv_len), max(conv_len)]``
            s_em (torch.Tensor): ``[batch_size, max(conv_len), max(conv_len)]``
            s_span (torch.Tensor): ``[n_arcs, max(ut_len)]``
            graphs (torch.Tensor): ``[batch_size, max(conv_len), max(conv_len)]``
            emotions (torch.Tensor): ``[batch_size, max(conv_len)]``
            spans (torch.Tensor): ``[batch_size, max(conv_len), max(conv_len), max(ut_len)]``
            ut_mask (torch.Tensor): ``[batch_size, max(conv_len)]``
        """
        uts = graphs[ut_mask]






