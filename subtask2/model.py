
import torch.nn as nn
from transformers import AutoModel
import torch
from torch.nn.utils.rnn import pad_sequence
from modules import *
from typing import List, Tuple, Optional
from utils import Config, to, expand_mask
from transformers.feature_extraction_utils import BatchFeature
from torchvision.transforms import Resize


class Subtask2Model(nn.Module):
    DECODER = ['ut_cause', 'ut_effect', 'em_cause', 'em_effect', 'ut_attn', 'em_attn']
    PARAMS = ['ut_embed_size', 'text_conf', 'spk_conf', 'img_conf', 'audio_conf', 'em_conf']

    def __init__(
            self,
            ut_embed_size: int, 
            device: str,
            text_conf: Config,
            spk_conf: Config,
            img_conf: Optional[Config], 
            audio_conf: Optional[Config],            
            em_conf: Config,
    ):
        super().__init__()
        self.word_embed = PretrainedEmbedding(**text_conf())
        text_conf.embed_size = self.word_embed.embed_size
        self.spk_embed = nn.Embedding(spk_conf.vocab_size, spk_conf.embed_size, spk_conf.pad_index).to(text_conf.device)
        
        ut_input_size = text_conf.embed_size + spk_conf.embed_size
        
        # add audio embeddings
        if img_conf is not None:
            self.img_embed = PretrainedImageEmbedding(**img_conf())
            img_conf.embed_size = self.img_embed.embed_size
            ut_input_size += img_conf.embed_size
        else:
            self.img_embed = nn.Identity()
            
        if audio_conf is not None:
            self.audio_embed = PretrainedAudioEmbedding(**audio_conf())
            ut_input_size += audio_conf.embed_size 
        else:
            self.audio_embed = nn.Identity()

        # decoder 
        self.ut_cause = FFN(ut_input_size, ut_embed_size)
        self.ut_effect = FFN(ut_input_size, ut_embed_size)
        self.ut_attn = Biaffine(n_in=ut_embed_size, n_out=2, bias_x=True, bias_y=True, dropout=0.1)

        self.em_cause = FFN(ut_input_size, ut_embed_size)
        self.em_effect = FFN(ut_input_size, ut_embed_size)
        self.em_attn = Biaffine(n_in=ut_embed_size, n_out=em_conf.vocab_size, dropout=0.1, bias_x=True, bias_y=True)

        # loss functions 
        self.criterion = nn.CrossEntropyLoss()
        self.device = device 
        
        for layer in self.DECODER:
            self.__setattr__(layer, getattr(self, layer).to(device))
        for name, value in locals().items():
            if name in self.PARAMS:
                self.__setattr__(name, value)


    def forward(
            self,
            words: torch.Tensor,
            speakers: torch.Tensor,
            frames: Optional[List[List[torch.Tensor]]],
            audios: Optional[List[BatchFeature]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        frames = to(self.img_conf.device, *frames) if self.img_conf is not None else None 
        audios = to(self.audio_conf.device, *audios) if self.audio_conf is not None else None
        ut_embed = self.encode(*to(self.text_conf.device, words, speakers), frames, audios)
        return self.decode(ut_embed)        
        

    def encode(
            self,
            words: torch.Tensor,
            speakers: torch.Tensor,
            frames: Optional[List[torch.Tensor]],
            audios: Optional[List[BatchFeature]]
    ) -> torch.Tensor:
        batch_size = words.shape[0]
        multimodal = []
        if frames is not None:
            multimodal.append(self.img_embed(frames))
        if audios is not None:
            multimodal.append(self.audio_embed(audios))
        word_embed = torch.stack([self.word_embed(words[i])[:, 0] for i in range(batch_size)], dim=0)
        spk_embed = self.spk_embed(speakers)
        
        # create utterance embeddings 
        ut_embed = torch.concat(to(self.device, word_embed, spk_embed, *multimodal), dim=-1)
        return ut_embed

    def decode(
        self, 
        ut_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # compute cause-effect representations
        ut_cause = self.ut_cause(ut_embed)
        ut_effect = self.ut_effect(ut_embed)
        s_ut = self.ut_attn(ut_cause, ut_effect).permute(0, 2, 3, 1)

        # compute emotion predictions using utterance representations
        em_cause = self.em_cause(ut_embed)
        em_effect = self.em_effect(ut_embed)
        s_em = self.em_attn(em_cause, em_effect).permute(0, 2, 3, 1)
        return s_ut, s_em

    def loss(
        self,
        s_ut: torch.Tensor, 
        s_em: torch.Tensor, 
        emotions: torch.Tensor,
        graphs: torch.Tensor, 
        pad_mask: torch.Tensor
    ) -> torch.Tensor:
        graphs, emotions, pad_mask = to(self.device, graphs, emotions, pad_mask)

        # compute utterance unlabeled cause-relation
        ut_mask = expand_mask(pad_mask)
        # ut_mask = expand_mask(pad_mask).to(self.device)
        ut_loss = self.criterion(s_ut[ut_mask], graphs[ut_mask].to(torch.long))

        # compute utterance labeled cause-relation gold
        ems = torch.zeros_like(graphs, dtype=torch.int32, device=graphs.device)
        b, cause, effect = (s_ut.argmax(-1).to(torch.bool) | graphs).nonzero(as_tuple=True)
        ems[b, cause, effect] = emotions[b, effect]
        em_loss = self.criterion(s_em[ut_mask], ems[ut_mask].to(torch.long))
        return ut_loss + em_loss


    def predict(
        self, 
        words: torch.Tensor,
        speakers: torch.Tensor,
        frames: List[List[torch.Tensor]],
        audios: List[BatchFeature],
        pad_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s_ut, s_em = self.forward(words, speakers, frames, audios)
        ut_preds, em_preds = s_ut.argmax(-1).to(torch.bool), s_em.mean(1)[:, :, 2:].argmax(-1) + 2
        ut_mask = expand_mask(pad_mask)
        ut_preds[~ut_mask] = False 
        return ut_preds, em_preds
        
        