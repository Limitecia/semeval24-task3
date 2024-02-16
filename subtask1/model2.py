
import torch.nn as nn
import torch
from modules import PretrainedEmbedding, Biaffine, FFN, LSTM
from torch.nn.utils.rnn import PackedSequence, pad_sequence
from utils import Config, to, expand_mask
from typing import Tuple, Optional
import numpy as np 

class Subtask1Model(nn.Module):
    EMBED_LAYERS = ['spk_embed']
    DECODER_LAYERS = ['em_attn', 'span_attn', 'ut_cause', 'ut_effect', 'em_cause', 'em_effect', 'ut_attn']

    def __init__(
        self,
        pretrained: str,
        word_config: Config,
        spk_config: Config,
        em_config: Config,
        ut_embed_size,
        num_heads: int = 1,
        finetune: bool = False,
        device: str = 'cuda:0',
        embed_device: str = 'cuda:1'
    ):
        super().__init__()
        self.word_embed = PretrainedEmbedding(pretrained, word_config.pad_index, finetune, embed_device)
        word_config.embed_size = self.word_embed.embed_size
        self.spk_embed = nn.Embedding(spk_config.vocab_size, spk_config.embed_size, spk_config.pad_index)
        self.em_pad_index = em_config.pad_index

        self.ut_cause = LSTM(word_config.embed_size + spk_config.embed_size, ut_embed_size, bidirectional=True, activation=nn.LeakyReLU())
        self.ut_effect = FFN(word_config.embed_size + spk_config.embed_size, ut_embed_size, nn.LeakyReLU())
        self.ut_attn = Biaffine(n_in=ut_embed_size, n_out=1, bias_x=True, bias_y=False, dropout=0.3)
        
        self.em_cause = LSTM(word_config.embed_size + spk_config.embed_size, ut_embed_size, bidirectional=True, activation=nn.LeakyReLU())
        self.em_effect = FFN(word_config.embed_size + spk_config.embed_size, em_config.embed_size, activation=nn.LeakyReLU())
        self.em_attn = Biaffine(n_in=ut_embed_size, n_out=em_config.vocab_size, bias_x=True, bias_y=False, dropout=0.3)
        # self.em_attn = nf.em_attn.out_proj.bias)
        
        self.span_attn = nn.MultiheadAttention(
            word_config.embed_size+spk_config.embed_size, num_heads=num_heads, dropout=0.1, batch_first=True, 
            kdim=ut_embed_size, vdim=ut_embed_size)
        nn.init.orthogonal_(self.span_attn.out_proj.weight)
        nn.init.zeros_(self.span_attn.out_proj.bias)

        self.criterion = nn.BCEWithLogitsLoss()
        self.em_criterion = nn.CrossEntropyLoss(weight=em_config.weights.to(device))
        self.span_criterion = nn.BCELoss()
        self.device = device
        self.embed_device = embed_device
        self.params = Config.from_class(locals())

        for layer in self.EMBED_LAYERS:
            self.__setattr__(layer, self.__getattr__(layer).to(embed_device))
        for layer in self.DECODER_LAYERS:
            self.__setattr__(layer, self.__getattr__(layer).to(device))



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
            s_ut (~torch.Tensor): ``[batch_size, max(conv_len), max(conv_len)]``
            s_em (~torch.Tensor): ``[batch_size, max(conv_len), max(conv_len), n_emotions]``
            s_span (~torch.Tensor): ``[batch_size,, max(conv_len), max(conv_len), max(ut_len)]``
        """
        word_embed, ut_embed = self.encode(words, speakers)
        return self.decode(word_embed, ut_embed)


    def encode(
            self, 
            words: torch.Tensor, 
            speakers: torch.Tensor, 
        ) -> torch.Tensor:
        r"""
        Encodes the inputs.
        
        Args:
            words (~torch.Tensor): ``[batch_size, max(conv_len), bos + max(ut_len), fix_len]``.
            speakers (~torch.Tensor): ``[batch_size, max(conv_len)]``.
        
        Returns:
            word_embed (~torch.Tensor): ``[batch_size, max(conv_len), bos + max(ut_len), word_embed_size]``.
        """
        # compute word embeddings
        words, speakers = to(self.embed_device, words, speakers)
        batch_size, *_ = words.shape[0], words.shape[1], words.shape[2]

        # word_embed ~ [batch_size, max(conv_len), max(ut_len), word_embed_size,]
        word_embed = torch.stack([
            self.word_embed(words[i]) 
            for i in range(batch_size)], dim=0).to(self.embed_device)

        # spk_embed ~ [batch_size, max(conv_len), spk_embed_size]
        speakers = torch.stack([s.unique(return_inverse=True)[1] for s in speakers], 0)
        spk_embed = self.spk_embed(speakers.to(self.embed_device))

        # ut_embed ~ [batch_size, max(conv_len), ut_embed_size]
        word_embed = torch.cat([word_embed, spk_embed.unsqueeze(-2).repeat(1, 1, word_embed.shape[-2], 1)], -1)
        ut_embed = word_embed[:, :, 0]
        return to(self.device, word_embed[:, :, 1:], ut_embed)

    def decode(
            self, 
            word_embed: torch.Tensor,
            ut_embed: torch.Tensor
            
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Decodes word contextualizations from the encoder.
        
        Args: 
            word_embed (~torch.Tensor): ``[batch_size, max(conv_len), bos + max(ut_len), word_embed_size]``
            
        Returns:
            s_ut (~torch.Tensor): ``[batch_size, max(conv_len), max(conv_len)]``
            s_em (~torch.Tensor): ``[batch_size, max(conv_len), max(conv_len), n_emotions]``
            s_span (~torch.Tensor): ``[batch_size,, max(conv_len), max(conv_len), max(ut_len)]``
        """
        
        
        # span_embed ~ [batch_size, max(conv_len), max(conv_len), max(ut_len)]
        ut_effect = self.ut_effect(ut_embed)
        em_effect = self.em_effect(ut_embed)
        ut_cause, em_cause, s_span = [], [], []
        for i in range(word_embed.shape[0]):
            # word ~ [conv_len, ut_len, word_embed_size]
            # ut ~ [conv_len, conv_len, ut_embed_size]
            ut_eff = ut_effect[i].unsqueeze(0).repeat(ut_effect.shape[1], 1, 1)
            word = word_embed[i]
            
            # word_cause ~ [conv_len, ut_len, word_embed_size]
            # scores ~ [conv_len, ut_len, conv_len]
            word_cause, scores = self.span_attn(word, ut_eff, ut_eff)
            scores = (scores.permute(0,2,1) - scores.min())/(scores.max() - scores.min())

            ut_cause.append(self.ut_cause(word_cause)[1])
            em_cause.append(self.em_cause(word_cause)[1])
            s_span.append(scores)
            
            
        ut_cause, s_span, em_cause = torch.stack(ut_cause, 0), torch.stack(s_span, 0), torch.stack(em_cause, 0)
            
        # compute cause-effect representations
        s_ut = self.ut_attn(ut_effect, ut_cause).permute(0, 2, 1)

        # compute emotion predictions using utterance representations
        s_em = self.em_attn(em_effect, em_cause).permute(0, 3, 2, 1)
        # s_em = self.em(self.em_attn(ut_cause, ut_cause, ut_embed, attn_mask=torch.sigmoid(s_ut))[0])
        
        # compute spans 
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        graphs, spans, emotions = to(self.device, graphs, spans, emotions)
        
        # compute utterance labeled cause-relation gold
        ut_mask = expand_mask(pad_mask).to(self.device) & (graphs | (s_ut > 0))
        ut_loss = self.criterion(s_ut[ut_mask], graphs[ut_mask].to(torch.float32))


        ems = torch.zeros_like(graphs, dtype=torch.int32, device=graphs.device)
        b, cause, effect = graphs.nonzero(as_tuple=True)
        ems[b, cause, effect] = emotions[b, effect]
        em_loss = self.em_criterion(s_em[ut_mask], ems[ut_mask].to(torch.long))

        # compute spans loss
        span_loss = self.span_criterion(s_span[span_mask].flatten(), spans[span_mask].flatten().to(torch.float32))
        return ut_loss, em_loss, span_loss

    def predict(
        self,
        words: torch.Tensor,
        speakers: torch.Tensor,
        pad_mask: torch.Tensor,
        span_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_ut, s_em, s_span = self(words, speakers, pad_mask)
        ut_preds, em_preds = s_ut > 0, s_em.mean(1)[:, :, 2:].argmax(-1) + 2
        ut_mask = expand_mask(pad_mask).to(self.device)
        ut_preds[~ut_mask] = False
        s_span[~span_mask] = 0
        s_span[~ut_preds] = 0
        span_preds = torch.zeros_like(s_span, dtype=torch.bool, device=s_span.device)
        if ut_preds.sum() > 0:
            span_preds[ut_preds] = torch.tensor(np.apply_along_axis(smooth, -1, s_span[ut_preds].cpu().numpy()), device=s_span.device)
        return ut_preds, em_preds, span_preds



def smooth(x: np.ndarray):
    mask = np.zeros_like(x, dtype=np.bool_)
    if (x > 0.5).sum() < 1:
        arg1 = np.argsort(x)[0]
        mask[(arg1-1):(arg1+1)] = True
    else:
        start, *_, end = (x > 0.5).flatten().nonzero()[0].tolist()
        mask[start:(end+1)] = True
    return mask
        
    
    