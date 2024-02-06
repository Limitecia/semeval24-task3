import torch.nn as nn
import torch

class FFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: nn.Module = nn.Identity()
    ):
        super().__init__()
        self.mlp = nn.Linear(in_features=in_features, out_features=out_features)
        self.act = activation
        self.out_features = out_features
        self.in_features = in_features
        self.out_features = out_features
        self.reset_parameters()

    def forward(self, x: torch.Tensor):
        if len(x.shape) > 2:
            return self.act(self.mlp(x.contiguous().view(-1, x.shape[-1]))).view(x.shape[0], x.shape[1], self.out_features)
        else:
            return self.act(self.mlp(x))


    def reset_parameters(self):
        nn.init.orthogonal_(self.mlp.weight)
        nn.init.zeros_(self.mlp.bias)