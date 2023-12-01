import torch
import torch.nn as nn
from transformers import AutoModel
from torch.nn.utils.rnn import pad_sequence
from typing import Optional
from modules.ffn import FFN

class PretrainedEmbedding(nn.Module):
    def __init__(self, pretrained: str, embed_size: int, pad_index: int, finetune):
        super().__init__()
        self.word_embed = AutoModel.from_pretrained(pretrained).requires_grad_(finetune)
        self.pad_index = pad_index
        self.ffn = FFN(in_features=self.word_embed.config.hidden_size, out_features=embed_size, activation=nn.LeakyReLU(0.1))
        self.max_len = self.word_embed.config.max_position_embeddings

    def forward(self, words: torch.Tensor) -> torch.Tensor:

        # words ~ [batch_size, pad(seq_len), fix_len]
        # mask ~ [batch_Size, pad(seq_len)]
        mask = (words != self.pad_index).sum(-1) > 0
        lens = mask.sum(-1).tolist()
        word_lens = (words[mask] != self.pad_index).sum(-1).tolist()

        # remove fix_len dimension to obtain flat ~ [batch_size, pad(seq_len) + fix_len]
        flat = words.flatten(-2,-1)
        attention = (flat != self.pad_index).to(torch.int32)

        x = []
        for i in range(0, flat.shape[1], self.max_len):
            x.append(self.word_embed(flat[:, i:(i+self.max_len)], attention_mask=attention[:, i:(i+self.max_len)]).last_hidden_state)  # ~ [batch_size, pad(seq_len) * fix_len, n_encoder_hidden]
        x = torch.cat(x, dim=1)

        # compute mean per word: xflat ~ [batch_size * pad(seq_len) * fix_len, embed_size)
        pad = x[0, -1]
        x_words = [i.mean(0) for i in x[attention.to(torch.bool)].split(word_lens)]
        x_seq = torch.stack([torch.stack([x_words.pop(0) for _ in range(l)] + [pad for _ in range(words.shape[1]-l)], dim=0) for l in lens], dim=0)

        embed = self.ffn(x_seq)
        return embed




if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModel
    from torch.nn.utils.rnn import pad_sequence
    tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-560m', padding_side='right')
    embed = PretrainedEmbedding('bigscience/bloom-560m', 100, tokenizer.pad_token_id, False)

    phrases = ['My dog is really cute', 'I could not attend to the last lesson since I had a medical appointment']
    fix_len = 4
    words = pad_sequence(
        [tokenizer(phrase.split(), padding='max_length', max_length=fix_len, add_special_tokens=True, truncation=True, return_tensors='pt')['input_ids'] for phrase in phrases],
        padding_value=tokenizer.pad_token_id, batch_first=True)

