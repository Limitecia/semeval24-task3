import torch, os
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoImageProcessor
from transformers.feature_extraction_utils import BatchFeature
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, List
from modules.ffn import FFN
from modules.lstm import LSTM
from utils import split, expand_mask
from torchvision.io import read_video
from torch.nn.functional import interpolate


class PretrainedEmbedding(nn.Module):
    def __init__(self, pretrained: str, pad_index: int, finetune: bool, device: str, embed_size: Optional[int] = None):
        super().__init__()
        self.word_embed = AutoModel.from_pretrained(pretrained).requires_grad_(finetune).to(device)
        self.pad_index = pad_index
        if embed_size:
            self.ffn = FFN(self.word_embed.config.hidden_size, embed_size).to(device)
            self.embed_size = embed_size
        else:
            self.ffn = nn.Identity()
            self.embed_size = self.word_embed.config.hidden_size
        self.max_len = self.word_embed.config.max_position_embeddings
        self.device = device

    def forward(self, words: torch.Tensor) -> torch.Tensor:
        # words ~ [batch_size, pad(seq_len), fix_len]
        # mask ~ [batch_size, pad(seq_len)]
        mask = (words != self.pad_index).sum(-1) > 0
        lens = mask.sum(-1).tolist()

        # flat ~ [batch_size, pad(seq_len)]
        fmask = words != self.pad_index
        flat = pad_sequence(words[fmask].split(fmask.sum((-2, -1)).tolist()), batch_first=True,
                            padding_value=self.pad_index)
        x = self.word_embed(flat, attention_mask=(flat != self.pad_index).to(torch.int32)).last_hidden_state

        word_lens = fmask.sum(-1).flatten()
        word_lens = word_lens[word_lens > 0].tolist()
        x = split([torch.mean(i, dim=0) for i in x[flat != self.pad_index].split(word_lens)],
                  [l for l in lens if l > 0])
        x_seq = pad_sequence([torch.stack(i, dim=0) for i in x], padding_value=0, batch_first=True)
        embed = self.ffn(x_seq.to(self.device))

        # padding is needed
        if words.shape[1] > embed.shape[1]:
            embed = torch.concat(
                [embed, torch.zeros(embed.shape[0], words.shape[1] - embed.shape[1], embed.shape[2]).to(self.device)],
                dim=1)
        if words.shape[0] > embed.shape[0]:
            embed = torch.concat(
                [embed, torch.zeros(words.shape[0] - embed.shape[0], embed.shape[1], embed.shape[2]).to(self.device)],
                dim=0)

        return embed



class PretrainedImageEmbedding(nn.Module):
    def __init__(
        self,
        pretrained: str, 
        device: str,
        finetune: bool,
        embed_size: Optional[int] = None,
    ):
        super().__init__()
        self.img_embed = AutoModel.from_pretrained(pretrained).to(device).requires_grad_(finetune)
        hidden_size = AutoConfig.from_pretrained(pretrained).hidden_size
        self.embed_size = embed_size or hidden_size
        self.frame = LSTM(hidden_size, self.embed_size)
        self.device = device
        self.to(device)
        
        
    def forward(self, batch: List[List[torch.Tensor]]) -> torch.Tensor:
        embed = []
        for frames in batch:
            emb = torch.stack([self.embed(x) for x in frames], 0)
            embed.append(emb)
        return pad_sequence(embed, True)
            
    
    def embed(self, frames: torch.Tensor) -> torch.Tensor:
        embed = self.img_embed(frames).last_hidden_state.flatten(0, 1)
        return self.frame(embed.unsqueeze(0))[1].squeeze(0)
    
    
    
    

class PretrainedAudioEmbedding(nn.Module):
    def __init__(
        self, 
        pretrained: str,
        finetune: bool,
        device: str, 
        embed_size: Optional[int] = None
    ):
        super().__init__()
        self.audio_embed = AutoModel.from_pretrained(pretrained).requires_grad_(finetune)
        self.device = device 
        hidden_size = self.audio_embed.config.hidden_size
        self.embed_size = embed_size or hidden_size
        self.proj = LSTM(hidden_size, embed_size)
        self.to(device)
        
    def forward(self, audios: List[BatchFeature]):
        embed = []
        for audio in audios:
            x = self.proj(self.audio_embed(**audio).last_hidden_state)[1]
            embed.append(x) 
        return pad_sequence(embed, True)
    
    

        
        
