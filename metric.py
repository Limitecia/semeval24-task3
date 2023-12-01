from __future__ import annotations
from typing import Optional
import torch

class Metric:
    ATTRIBUTES = []
    def __init__(self):
        for attr in self.ATTRIBUTES:
            self.__setattr__(attr, 0.0)

    def __add__(self, other: Metric) -> Metric:
        for attr in self.ATTRIBUTES:
            self.__setattr__(attr, getattr(self, attr) + getattr(other, attr))


class Subtask1Metric(Metric):
    ATTRIBUTES = ['n', '_etp', '_stp', '_loss']
    METRICS = ['F1', 'wF1', 'pF1', 'loss']

    def __init__(
        self,
        loss: Optional[torch.Tensor] = None,

    ):
        super().__init__()

