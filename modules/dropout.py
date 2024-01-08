from __future__ import annotations
import torch
import torch.nn as nn

class SharedDropout(nn.Module):

    def __init__(self, p: float = 0.5, batch_first: bool = True) -> SharedDropout:
        super().__init__()

        self.p = p
        self.batch_first = batch_first

    def __repr__(self):
        s = f"p={self.p}"
        if self.batch_first:
            s += f", batch_first={self.batch_first}"
        return f"{self.__class__.__name__}({s})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        return x * self.get_mask(x[:, 0], self.p).unsqueeze(1) if self.batch_first else self.get_mask(x[0], self.p)

    @staticmethod
    def get_mask(x: torch.Tensor, p: float) -> torch.FloatTensor:
        return x.new_empty(x.shape).bernoulli_(1 - p) / (1 - p)