
import torch.nn as nn
import torch
from modules import PretrainedEmbedding, LSTM, Biaffine, FFN
from torch.nn.utils.rnn import PackedSequence
from utils import Config
from typing import Tuple

class EmotionCausalMdoel(nn.Module):
    def __init__(
        self,
        pretrained: str,
        word_config: Config,
        ut_config: Config,
        speaker_config: Config,
        emotion_config: Config,
        pos_embed_size: int,
        avg_span_len: int
    ):
        self.word_embed = PretrainedEmbedding(pretrained, word_config.embed_size, word_config.pad_index)
        self.ut_embed = LSTM(
            input_size=self.word_config.embed_size,
            hidden_size=self.ut_config.embed_size,
            activation=nn.LeakyReLU(0.1))
        self.spk_embed = nn.Embedding(
            num_embeddings=speaker_config.vocab_size, embedding_dim=speaker_config.embed_size, padding_idx=speaker_config.pad_index)
        self.em_embed = nn.Embedding(
            num_embeddings=emotion_config.vocab_size, embedding_dim=emotion_config.embed_size, padding_idx=emotion_config.pad_index
        )

        self.ut_cause = FFN(in_features=ut_config.embed_size+speaker_config.embed_size,
                            out_features=ut_config.embed_size, activation=nn.LeakyReLU(0.1))
        self.ut_effect = FFN(in_features=ut_config.embed_size+speaker_config.embed_size,
                             out_features=ut_config.embed_size, activation=nn.LeakyReLU(0.1))
        self.em_cause = FFN(in_features=ut_config.embed_size+speaker_config.embed_size,
                            out_features=ut_config.embed_size, activation=nn.LeakyReLU(0.1))
        self.em_effect = FFN(in_features=ut_config.embed_size+speaker_config.embed_size,
                             out_features=ut_config.embed_size, activation=nn.LeakyReLU(0.1))

        self.ut_attn = Biaffine(n_in=ut_config.embed_size, n_out=1, bias_x=True, bias_y=False)
        self.em_attn = Biaffine(n_in=ut_config.embed_size, n_out=emotion_config.vocab_size, bias_x=True, bias_y=True)

        self.span = nn.Conv1d(
            in_channels=word_config.embed_size + pos_embed_size + emotion_config.embed_size + ut_config.embed_size,
            out_channels=1, kernel_size=avg_span_len, activation=nn.Sigmoid()
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        words: torch.Tensor,
        speakers: torch.Tensor,
        causes: torch.Tensor,
        emotions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Args:
            words (torch.Tensor): ``[batch_size, pad(conv_len), pad(ut_len), fix_len]``.
            speakers (torch.Tensor): ``[batch_size, pad(conv_len)]``.
            emotions (torch.Tensor): ``[batch_size, pad(conv_len)]``
            causes (torch.Tensor): ``[batch_size, pad(conv_len), pad(conv_len)]``

        Returns:
            ~torch.Tensor, ~torch.Tensor, ~torch.Tensor
            - s_ut: ``[batch_size, pad(conv_len), pad(conv_len)]``
            - s_em: ``[batch_size, pad(conv_len), pad(conv_len), n_emotions]``
            - s_span: ``[batch_size, pad(conv_len), pad(ut_len)]``
        """
        # compute word embeddings
        word_embed = self.word_embed(words)

        # compute utterance embeddings
        spk_embed = self.speaker(speakers)
        ut_embed = self.ut_embed(word_embed)
        ut_embed = torch.concat([ut_embed, spk_embed], dim=-1)

        # compute cause-effect representations
        ut_cause = self.ut_cause(ut_embed)          # [batch_size, pad(conv_len), ut_embed_size]
        ut_effect = self.ut_effect(ut_embed)        # [batch_size, pad(conv_len), ut_embed_size]
        em_cause = self.em_cause(ut_embed)          # [batch_size, pad(conv_len), ut_embed_size]
        em_effect = self.em_effect(ut_embed)        # [batch_size, pad(conv_len), ut_embed_size]

        # Biaffine utterance step
        s_ut = self.ut_attn(ut_effect, ut_cause)
        s_em = self.em_attn(em_effect, em_cause)

        # Spans prediction

        return s_ut, s_em



    def loss(
        self,
        s_ut: torch.Tensor,
        s_em: torch.Tensor,
        ems: torch.Tensor,
        ut_mask: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Args:
            s_ut (torch.Tensor): ``[batch_size, pad(conv_len), pad(conv_len)]``
            s_em (torch.Tensor): ``[batch_size, pad(conv_len), pad(conv_len)]``
            ems (torch.Tensor): ``[batch_size, pad(conv_len), pad(conv_len)]``

        """



