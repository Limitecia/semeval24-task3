
import torch.nn as nn
import torch
from modules import PretrainedEmbedding, LSTM, Biaffine, FFN
from torch.nn.utils.rnn import PackedSequence, pad_sequence
from utils import Config, to
from typing import Tuple, Optional

class EmotionCausalModel(nn.Module):
    EMBED_LAYERS = ['ut_embed', 'spk_embed', 'em_embed']
    DECODER_LAYERS = ['ut_cause', 'ut_effect','ut_attn', 'em', 'em_cls', 'span']

    def __init__(
        self,
        pretrained: str,
        word_config: Config,
        spk_config: Config,
        em_config: Config,
        ut_embed_size,
        finetune: bool = False,
        device: str = 'cuda:0',
        embed_device: str = 'cuda:1'
    ):
        super().__init__()
        self.word_embed = PretrainedEmbedding(pretrained, word_config.embed_size, word_config.pad_index, finetune=finetune, device=embed_device)
        self.ut_embed = nn.LSTM(input_size=word_config.embed_size, hidden_size=word_config.embed_size//2,
                                bidirectional=True, dropout=0.33, num_layers=4, batch_first=True)
        self.spk_embed = nn.Embedding(
            num_embeddings=spk_config.vocab_size, embedding_dim=spk_config.embed_size, padding_idx=spk_config.pad_index)
        self.em_embed = LSTM(
            input_size=word_config.embed_size, hidden_size=em_config.embed_size, num_layers=3, 
            bidirectional=True, activation=nn.LeakyReLU(), dropout=0.33
        )
        self.em_pad_index = em_config.pad_index

        self.ut_cause = FFN(in_features=word_config.embed_size+spk_config.embed_size, out_features=ut_embed_size, activation=nn.LeakyReLU())
        self.ut_effect = FFN(in_features=word_config.embed_size+spk_config.embed_size, out_features=ut_embed_size, activation=nn.LeakyReLU())
        self.ut_attn = Biaffine(n_in=ut_embed_size, n_out=1, bias_x=True, bias_y=False)


        self.em = nn.MultiheadAttention(embed_dim=em_config.embed_size, num_heads=1, dropout=0.1, batch_first=True, kdim=ut_embed_size, vdim=ut_embed_size)
        self.em_cls = FFN(in_features=em_config.embed_size, out_features=em_config.vocab_size, activation=nn.Softmax(-1))
        self.span = LSTM(
            input_size=word_config.embed_size + em_config.embed_size,
            hidden_size=word_config.embed_size, output_size=1, activation=nn.LeakyReLU(),
            bidirectional=True, num_layers=4, dropout=0.33
        )


        self.ut_loss = nn.BCEWithLogitsLoss()
        self.em_loss = nn.CrossEntropyLoss()
        self.device = device
        self.embed_device = embed_device

        for layer in self.EMBED_LAYERS:
            self.__setattr__(layer, self.__getattr__(layer).to(embed_device))
        for layer in self.DECODER_LAYERS:
            self.__setattr__(layer, self.__getattr__(layer).to(device))



    def forward(
        self,
        words: torch.Tensor,
        speakers: torch.Tensor,
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
        graphs, spans = to([graphs, spans], self.device)
        word_embed, ut_embed, em_embed = self.encode(words, speakers)
        s_ut, s_em, s_span = self.decode(word_embed, ut_embed, em_embed, graphs, spans)
        return s_ut, s_em, s_span


    def encode(
            self, 
            words: torch.Tensor, 
            speakers: torch.Tensor, 
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # compute word embeddings
        words, speakers = words.cuda(), speakers.to(self.embed_device)
        batch_size, *_ = words.shape[0], words.shape[1], words.shape[2]

        # word_embed ~ [batch_size, max(conv_len), max(ut_len), word_embed_size]
        word_embed = torch.stack([
            self.word_embed(words[i]) 
            for i in range(batch_size)], dim=0).to(self.embed_device)

        # spk_embed ~ [batch_size, max(conv_len), spk_embed_size]
        spk_embed = self.spk_embed(speakers)

        # ut_embed ~ [batch_size, max(conv_len), ut_embed_size]
        # ut_embed = torch.stack([word_embed[i,:, 0] for i in range(batch_size)], dim=0)
        ut_embed = torch.stack([
            self.ut_embed(word_embed[i])[0][:, 0] for i in range(batch_size)],
            dim=0
        )
        ut_embed = torch.concat([ut_embed, spk_embed], dim=-1)
        em_embed = torch.stack([self.em_embed(word_embed[i])[:, 0] for i in range(batch_size)], dim=0)
        return word_embed.to(self.device), ut_embed.to(self.device), em_embed.to(self.device)

    def decode(
            self, 
            word_embed: torch.Tensor, 
            ut_embed: torch.Tensor, 
            em_embed: torch.Tensor,
            graphs: torch.Tensor,
            spans: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = word_embed.shape[0]

        # compute cause-effect representations
        ut_cause = self.ut_cause(ut_embed)          # [batch_size, pad(conv_len), ut_embed_size]
        ut_effect = self.ut_effect(ut_embed)        # [batch_size, pad(conv_len), ut_embed_size]
        s_ut = self.ut_attn(ut_effect, ut_cause)

        # compute emotion predictions using utterance representations
        s_em, _ = self.em(em_embed, ut_cause, ut_cause, attn_mask=graphs)
        s_em = self.em_cls(s_em)

        # compute spans 
        # cause_mask = (s_ut > 0) | graphs
        # s_span = torch.zeros_like(spans, dtype=torch.float32, device=self.device)
        # if cause_mask.sum() > 0:
        #     b, cause, effect = cause_mask.nonzero(as_tuple=True)
        #     em_effect = em_embed[b, effect]
        #     word_cause = word_embed[b, cause, 1:]
        #     span_embed = torch.concat([word_cause, em_effect.unsqueeze(1).repeat(1, word_cause.shape[1], 1)], dim=-1).to(self.device)
        #     span_embed = torch.concat([self.span(span_embed[i:(i+batch_size)]).squeeze(-1) for i in range(0, len(b), batch_size)], dim=0)
        #     s_span[b, cause, effect] = span_embed
        return s_ut, s_em, spans



    def loss(
        self,
        s_ut: torch.Tensor,
        s_em: torch.Tensor,
        s_span: torch.Tensor,
        emotions: torch.Tensor, 
        graphs: torch.Tensor,
        spans: torch.Tensor,
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
        graphs, spans, emotions = to([graphs, spans, emotions], self.device)

        # compute utterance unlabeled cause-relation
        ut_mask = (s_ut > 0) | graphs
        ut_loss = self.ut_loss(s_ut[ut_mask], graphs[ut_mask].to(torch.float32))

        # compute utterance labeled cause-relation gold
        em_loss = self.em_loss(s_em[pad_mask], emotions[pad_mask].to(torch.long))

        # compute spans loss
        # span_loss = self.ut_loss(s_span[ut_mask].flatten(), spans[ut_mask].flatten().to(torch.float32))

        return ut_loss

    def predict(
        self,
        words: torch.Tensor,
        speakers: torch.Tensor,
        pad_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Args:
            words (torch.Tensor): ``[batch_size, max(conv_len), max(ut_len), fix_len]``
            speakers (torch.Tensor): ``[batch_size, max(conv_len)]``
            pad_mask (torch.Tensor): ``[batch_size, max(conv_len)]``
        Returns:
            em_preds (torch.Tensor): ``[batch_size, max(conv_len), max(conv_len)]``
            s_span (torch.Tensor): ``[batch_size, max(conv_len), max(conv_len), max(ut_len)]``
        """
        batch_size, max_conv_len, max_ut_len, _ = words.shape 
        word_embed, ut_embed, em_embed = to(self.encode(words, speakers), self.device)
        ut_cause = self.ut_cause(ut_embed)         
        ut_effect = self.ut_effect(ut_embed)       
        s_ut = self.ut_attn(ut_effect, ut_cause)

        ut_preds = s_ut > 0
        ut_preds[~pad_mask] = False

        s_em, _ = self.em(em_embed, ut_cause, ut_cause, attn_mask=ut_preds)
        s_em = self.em_cls(s_em)

        # compute spans
        s_span = torch.zeros((batch_size, max_conv_len, max_conv_len, max_ut_len-1), dtype=torch.float32, device=self.device)
        if ut_preds.sum() > 0:
            b, cause, effect = ut_preds.nonzero(as_tuple=True)
            em_effect = em_embed[b, effect]
            word_cause = word_embed[b, cause, 1:]
            span_embed = torch.concat([word_cause, em_effect.unsqueeze(1).repeat(1, word_cause.shape[1], 1)], dim=-1).to(self.device)
            span_embed = torch.concat([self.span(span_embed[i:(i+batch_size)]).squeeze(-1) for i in range(0, len(b), batch_size)], dim=0)
            s_span[b, cause, effect] = span_embed

        return ut_preds, s_em.argmax(-1), s_span > 0

