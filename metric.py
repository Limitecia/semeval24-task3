from __future__ import annotations
from typing import Optional
import torch



class Subtask1Metric:
    ATTRIBUTES = ['n', 'count', '_etp', '_pred', '_gold', '_stp', '_spred', '_sgold', '_loss']
    METRICS = ['fs', 'pfs', 'loss']
    eps = 1e-12

    def __init__(
        self,
        loss: Optional[torch.Tensor] = None,
        em_preds: Optional[torch.Tensor] = None,
        span_preds: Optional[torch.Tensor] = None,
        ems: Optional[torch.Tensor] = None,
        spans: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        pad_index: Optional[int] = None
    ):
        for attr in self.ATTRIBUTES:
            self.__setattr__(attr, 0.0)

        if loss is not None:
            self.__call__(loss, em_preds, span_preds, ems, spans, pad_mask, pad_index)

    def __add__(self, other: Subtask1Metric) -> Subtask1Metric:
        for attr in self.ATTRIBUTES:
            self.__setattr__(attr, getattr(self, attr) + getattr(other, attr))
        return self

    def __call__(
        self,
        loss: torch.Tensor,
        em_preds: torch.Tensor,
        span_preds: torch.Tensor,
        ems: torch.Tensor,
        spans: torch.Tensor,
        pad_mask: torch.Tensor,
        pad_index: int
    ) -> Subtask1Metric:

        self.count += 1
        self.n += em_preds.shape[0]
        self._loss += loss

        ut_mask = (em_preds == ems)
        ut_mask[(em_preds == pad_index) | (ems == pad_index)] = False
        self._etp += ut_mask.sum()
        self._pred += (em_preds[pad_mask] != pad_index).sum()
        self._gold += (ems[pad_mask] != pad_index).sum()

        ut_mask
        if ut_mask.sum() > 0:
            self._stp += (span_preds[ut_mask] & spans[ut_mask]).sum()
            self._spred += span_preds[pad_mask].sum()
            self._sgold += spans[pad_mask].sum()

        for attr in self.ATTRIBUTES:
            res = getattr(self, attr)
            if isinstance(res, torch.Tensor):
                self.__setattr__(attr, res.item())

        return self


    @property
    def fs(self) -> float:
        rec = self._etp/(self._gold + self.eps)
        prec = self._etp/(self._pred + self.eps)
        f1 = (2*rec*prec)/(rec+prec + self.eps)
        return f1

    @property
    def loss(self):
        return self._loss/self.n

    @property
    def pfs(self):
        rec = self._stp/(self._sgold + self.eps)
        prec = self._stp/(self._spred + self.eps)
        f1 = (2*rec*prec)/(rec+prec + self.eps)
        return f1


    def __repr__(self):
        return ' '.join(f'{name}={round(getattr(self, name), 2)}' for name in self.METRICS)