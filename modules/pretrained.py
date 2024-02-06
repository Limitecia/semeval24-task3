import torch
import torch.nn as nn
from transformers import AutoModel
from torch.nn.utils.rnn import pad_sequence
from typing import Optional
from modules.ffn import FFN
from utils import split, expand_mask

class PretrainedEmbedding(nn.Module):
    def __init__(self, pretrained: str, embed_size: Optional[int], pad_index: int, finetune: bool, device: str):
        super().__init__()
        self.word_embed = AutoModel.from_pretrained(pretrained).requires_grad_(finetune).to(device)
        self.pad_index = pad_index
        if embed_size:
            self.ffn = FFN(in_features=self.word_embed.config.hidden_size, out_features=embed_size, activation=nn.LeakyReLU()).to(device)
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
        flat = pad_sequence(words[fmask].split(fmask.sum((-2,-1)).tolist()), batch_first=True, padding_value=self.pad_index)
        x = self.word_embed(flat, attention_mask=(flat != self.pad_index).to(torch.int32)).last_hidden_state

        word_lens = fmask.sum(-1).flatten()
        word_lens = word_lens[word_lens > 0].tolist()
        x = split([torch.mean(i, dim=0) for i in x[flat != self.pad_index].split(word_lens)], [l for l in lens if l > 0])
        x_seq = pad_sequence([torch.stack(i, dim=0) for i in x], padding_value=0, batch_first=True)
        embed = self.ffn(x_seq.to(self.device))
        
        # padding is needed 
        if words.shape[1] > embed.shape[1]:
            embed = torch.concat([embed, torch.zeros(embed.shape[0], words.shape[1]-embed.shape[1], embed.shape[2]).to(self.device)], dim=1)
        if words.shape[0] > embed.shape[0]:
            embed = torch.concat([embed, torch.zeros(words.shape[0]-embed.shape[0], embed.shape[1], embed.shape[2]).to(self.device)], dim=0)
            
        return embed

