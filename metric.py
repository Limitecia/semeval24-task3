from __future__ import annotations
from typing import Optional, Union, List
import torch, os
from data import Conversation
from evaluate import *


class Metric:
    METRICS = [] 
    KEY_METRICS = []
    
    def __init__(
        self,
        path: Optional[str] = None,
    ):
        for attr in self.METRICS:
            self.__setattr__(attr, 0.0)
        
        if path is not None:
            self(path)
    
    def __add__(self, other: Metric) -> Metric:
        for attr in self.METRICS:
            self.__setattr__(attr, getattr(self, attr) + getattr(other, attr))
        return self
    
    def __call__(self, path: str) -> Subtask1Metric:
        raise NotImplementedError
    
    def __repr__(self):
        return f', '.join(f'{name.upper()}={round(getattr(self, name)*100, 2)}' for name in self.METRICS)
    
    def improves(self, other: Metric) -> bool:
        assert all(k1 == k2 for k1, k2 in zip(self.KEY_METRICS, other.KEY_METRICS)) 
        return any(getattr(self, k1) > getattr(other, k2) for k1, k2 in zip(self.KEY_METRICS, other.KEY_METRICS))

        
class Subtask1Metric(Metric):
    METRICS = ['wsP', 'wsR', 'wsF1', 'wpP', 'wpR', 'wpF1', 'sP', 'sR', 'sF1', 'pP', 'pR', 'pF1']
    KEY_METRICS = ['wsF1', 'wpF1', 'pF1']


    def __call__(
        self,
        path: str,
    ) -> Subtask1Metric:
        gold_file = os.path.join(path, 'ref', 'Subtask_1_gold.json')
        pred_file = os.path.join(path, 'res', 'Subtask_1_pred.json')
        pred_data = get_json_data(pred_file)
        gold_data = get_json_data(gold_file)
        score_list, score_list_1 = evaluate_1_2(pred_data, gold_data)
        self.wsP, self.wsR, self.wsF1 = score_list[0:3]
        self.wpP, self.wpR, self.wpF1 = score_list_1[0:3]
        self.sP, self.sR, self.sF1 = score_list[3:6]
        self.pP, self.pR, self.pF1 = score_list_1[3:6]
        return self



class Subtask2Metric(Metric):
    METRICS = ['P', 'R', 'F1', 'WP', 'WR', 'WF1']
    KEY_METRICS = ['F1', 'WF1']

    def __call__(
        self,
        path: str,
    ) -> Subtask2Metric:
        gold_file = os.path.join(path, 'ref', 'Subtask_2_gold.json')
        pred_file = os.path.join(path, 'res', 'Subtask_2_pred.json')
        pred_data = get_json_data(pred_file)
        gold_data = get_json_data(gold_file)
        score_list = evaluate_2_2(pred_data, gold_data)
        self.P, self.R, self.F1 = score_list[0:3]
        self.WP, self.WR, self.WF1 = score_list[3:6]
        return self
