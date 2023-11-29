import torch.nn as nn
from modules.ffn import FFN
import torch
from typing import Optional

class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 1,
        bidirectional: bool = True,
        activation: nn.Module = nn.Identity(),
        dropout: float = 0.0
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=bidirectional, dropout=dropout, batch_first=True, bias=True)
        if output_size is None:
            self.ffn = activation
        else:
            self.ffn = FFN(in_features=hidden_size, out_features=output_size, activation=activation)

    def forward(self, x: torch.Tensor):
        h, _ = self.lstm(x)
        y = self.ffn(h)
        return y

