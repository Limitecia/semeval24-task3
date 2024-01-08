from __future__ import annotations
from typing import Optional
import torch



class Subtask1Metric:
    ATTRIBUTES = ['n', 'count', '_utp', '_upred', '_ugold', '_er', '_ep', '_ef', '_stp', '_spred', '_sgold', '_loss']
    METRICS = ['ur', 'up', 'uf', 'er', 'ep', 'ef', 'pf']
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

        # compute metrics for utterance mask
        ut_preds, uts = em_preds != pad_index, ems != pad_index
        self._utp += (ut_preds & uts).sum()
        self._upred += ut_preds.sum()
        self._ugold += uts.sum()

        # compute metrics for the emotions
        classes, counts = ems[ems != pad_index].unique(return_counts=True)
        etp, epred, egold = torch.zeros_like(classes), torch.zeros_like(classes), torch.zeros_like(classes)
        for i, c in enumerate(classes):
            etp[i] = ((em_preds == c) & (ems == c)).sum()
            epred[i] = (em_preds == c).sum()
            egold[i] = (ems == c).sum()
        weights = counts/counts.sum()
        er = etp/egold
        ep = torch.where(epred == 0, 0, etp/epred)
        self._ef += torch.dot(weights, torch.where(er+ep == 0, 0, (2*er*ep)/(er+ep)))
        self._er += torch.dot(weights, er)
        self._ep += torch.dot(weights, ep)

        # compute metrics for spans
        em_mask = (em_preds == ems) & (em_preds != pad_index) & (ems != pad_index)
        if em_mask.sum() > 0:
            self._stp += (span_preds[em_mask] & spans[em_mask]).sum()
            self._spred += span_preds[pad_mask].sum()
            self._sgold += spans[pad_mask].sum()

        for attr in self.ATTRIBUTES:
            res = getattr(self, attr)
            if isinstance(res, torch.Tensor):
                self.__setattr__(attr, res.item())

        return self

    @property
    def ep(self):
        return self._ep/self.count

    @property
    def er(self):
        return self._er/self.count

    @property
    def ef(self):
        return self._ef/self.count

    @property
    def up(self):
        return self._utp/(self._upred + self.eps)

    @property
    def ur(self):
        return self._utp/self._ugold

    @property
    def uf(self):
        return (2*self.up*self.ur)/(self.ur + self.up + self.eps)

    @property
    def loss(self):
        return self._loss/self.count

    @property
    def pf(self):
        rec = self._stp/(self._sgold + self.eps)
        prec = self._stp/(self._spred + self.eps)
        f1 = (2*rec*prec)/(rec+prec + self.eps)
        return f1


    def __repr__(self):
        return f'loss={round(self.loss,2)}, ' + ', '.join(f'{name.upper()}={round(getattr(self, name)*100, 2)}' for name in self.METRICS)