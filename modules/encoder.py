import torch.nn as nn
from typing import Optional, List, Union
from modules import PretrainedEmbedding, FFN, LSTM
import torch
from utils import Config

class Encoder(nn.Module):
    CONTEXTUALIZERS = [None, 'lstm']

    def __init__(
        self,
        vocab_size: int,
        pad_index: int,
        feats: List[str],
        n_feats: List[int],
        feats_pad: List[int],
        feats_size: Union[List[int], int] = 50,
        pretrained: Optional[str] = None,
        finetune: bool = False,
        embed_size: int = 200,
        context: str = None,
        bidirectional: bool = True,
        dropout: float = 0.2,
        num_layers: int = 2,
        **kwargs
    ):
        super().__init__()
        self.args = Config.from_class(locals().copy())

        self.pretrained = pretrained
        if pretrained:
            self._embed = PretrainedEmbedding(pretrained, embed_size, pad_index, finetune)
            self.context = nn.Identity()
        else:
            self._embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
            assert context in Encoder.CONTEXTUALIZERS, f'The contextualizer <{context}> is not available'
            if context is None:
                self.context = nn.Identity()
            elif context == 'lstm':
                self.context = LSTM(
                    input_size=embed_size, hidden_size=embed_size//2 if bidirectional else embed_size,
                    output_size=None, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)

        feats_size = [feats_size for _ in feats] if isinstance(feats_size, int) else feats_size
        for feat, n_feat, feat_size, feat_pad in zip(feats, n_feats, feats_size, feats_pad):
            self.__setattr__(feat, nn.Embedding(num_embeddings=n_feat, embedding_dim=feat_size))
        self.feats = feats
        self.pad_index = pad_index

        self.ffn = FFN(in_features=embed_size+sum(feats_size), out_features=embed_size) if len(feats) > 0 else nn.Identity()


    def embed(self, words: torch.Tensor, feats: List[torch.Tensor]) -> torch.Tensor:
        if self.pretrained:
            return self._embed(words)
        else:
            embeds = self._embed(words)
            if len(self.feats) > 0:
                efeats = [getattr(self, feat)(f) for feat, f in zip(self.feats, feats)]
                feat_embeds = torch.cat([embeds] + efeats, dim=-1)
                embeds = self.ffn(feat_embeds)
            return embeds

    def forward(self, words: torch.Tensor, feats: List[torch.Tensor]) -> torch.Tensor:
        assert len(feats) == len(self.feats), 'Number of provided features incorrect'
        embeds = self.embed(words, feats)
        embeds = self.context(embeds)
        return embeds
