from __future__ import annotations
from typing import Optional
from evaluate import get_json_data, evaluate_1_2, evaluate_2_2


class Metric:
    METRICS = [] 
    KEY_METRICS = []
    
    def __init__(
        self,
        pred: Optional[str] = None,
        gold: Optional[str] = None
    ):
        for attr in self.METRICS:
            self.__setattr__(attr, 0.0)
        
        if pred and gold:
            self(pred, gold)
    
    def __add__(self, other: Metric) -> Metric:
        for attr in self.METRICS:
            self.__setattr__(attr, getattr(self, attr) + getattr(other, attr))
        return self
    
    def __call__(self, pred: str, gold: str) -> Subtask1Metric:
        raise NotImplementedError
    
    def __repr__(self):
        return f', '.join(f'{name.upper()}={round(getattr(self, name)*100, 2)}' for name in self.METRICS)
    
    def improves(self, other: Metric) -> bool:
        assert all(k1 == k2 for k1, k2 in zip(self.KEY_METRICS, other.KEY_METRICS)) 
        return any(getattr(self, k1) > getattr(other, k2) for k1, k2 in zip(self.KEY_METRICS, other.KEY_METRICS))

        
class Subtask1Metric(Metric):
    METRICS = ['WSP', 'WSR', 'WSF', 'WPP', 'WPR', 'WPF', 'SP', 'SR', 'SF', 'PP', 'PR', 'PF']
    KEY_METRICS = ['WSF', 'WPF', 'PF']


    def __call__(self, pred: str, gold: str) -> Subtask1Metric:
        pred_data = get_json_data(pred)
        gold_data = get_json_data(gold)
        score_list, score_list_1 = evaluate_1_2(pred_data, gold_data)
        self.WSP, self.WSR, self.WSF = score_list[0:3]
        self.WPP, self.WPR, self.WPF = score_list_1[0:3]
        self.SP, self.SR, self.SF = score_list[3:6]
        self.PP, self.PR, self.PF = score_list_1[3:6]
        return self



class Subtask2Metric(Metric):
    METRICS = ['P', 'R', 'F', 'WP', 'WR', 'WF']
    KEY_METRICS = ['F', 'WF']

    def __call__(self, pred: str, gold: str) -> Subtask2Metric:
        pred_data = get_json_data(pred)
        gold_data = get_json_data(gold)
        score_list = evaluate_2_2(pred_data, gold_data)
        self.P, self.R, self.F = score_list[0:3]
        self.WP, self.WR, self.WF = score_list[3:6]
        return self
