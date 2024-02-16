from __future__ import annotations
from data.relation import CauseRelation
from data.utterance import Utterance
from typing import List, Dict, Optional 
import torch
import numpy as np

class Conversation:
    NO_CAUSE = '_'
    SEP = '_'

    def __init__(self, id: int, uts: List[Utterance], pairs: List[CauseRelation], subtask: int):
        self.id = id
        self.uts = uts
        self.pairs = pairs
        self.GRAPH = None
        self.SPAN = None
        self.build(subtask)
        self.subtask = subtask

    def build(self, subtask: int):
        self.GRAPH = torch.zeros(len(self), len(self), dtype=torch.bool)
        self.SPAN = torch.zeros(len(self), len(self), max(map(len, self.uts)))
        rem = []
        for i, pair in enumerate(self.pairs):
            if self.GRAPH[pair.CAUSE.ID - 1, pair.EFFECT.ID - 1]:
                rem.append(i)
                continue 
            self.GRAPH[pair.CAUSE.ID - 1, pair.EFFECT.ID - 1] = True
            if subtask != 2:
                self.SPAN[pair.CAUSE.ID - 1, pair.EFFECT.ID -1][:len(pair.SPAN)] = pair.SPAN.clone()
        self.pairs = [pair for i, pair in enumerate(self.pairs) if i not in rem]

        if subtask != 2:
            for ut in self.uts:
                self.SPAN[ut.ID - 1, :, len(ut):] = -1
        for field in Utterance.FIELDS:
            self.__setattr__(field, [getattr(ut, field) for ut in self.uts])
            
    @classmethod
    def from_dict(cls, data: dict, subtask: int, video_folder: Optional[str] = None) -> Conversation:
        id = data.pop('conversation_ID')
        conversation = data.pop('conversation')

        # extract uts
        uts = [Utterance.from_dict(ut, video_folder) for ut in conversation]


        # extract cause-relation pairs
        if 'emotion-cause_pairs' not in data.keys():
            pairs = []
        else:
            pairs = data.pop('emotion-cause_pairs')
            pairs = [(*pair[0].split(Conversation.SEP), *pair[1].split(Conversation.SEP)) for pair in pairs]
            if subtask == 2:
                pairs = [CauseRelation(CAUSE=uts[int(c) - 1], EFFECT=uts[int(e) - 1], SPAN=None) for e, _, c in pairs]
            else:
                pairs = [CauseRelation(CAUSE=uts[int(c) - 1], EFFECT=uts[int(e) - 1], SPAN=span) for e, _, c, span in pairs]
        return Conversation(id, uts, pairs, subtask)

    def __len__(self):
        return len(self.uts)
    
    def to_dict(self, submission: bool):
        rels = [rel.to_list(not submission) for rel in self.pairs]
        rels = list(filter(lambda x: x is not None, rels))
        return {'conversation_ID': self.id, 'conversation': [ut.to_dict() for ut in self.uts], 
                'emotion-cause_pairs': rels} 

    @property
    def span_lens(self) -> List[int]:
        return [pair.span_len for pair in self.pairs]
    
    @property
    def lens(self) -> List[int]:
        return [len(ut) for ut in self.uts]
    
    @property
    def max_len(self) -> int:
        return max(map(len, self.uts))
    
    @property 
    def emotions(self) -> Dict[str, int]:
        ems = dict()
        for pair in self.pairs:
            try:
                ems[pair.EMOTION] += 1 
            except:
                ems[pair.EMOTION] = 1
        return ems 


    def update1(self, graph: torch.Tensor, ems: List[str], spans: torch.Tensor):
        graph, spans = graph.cpu(), spans.cpu()
        uts = [ut.copy() for ut in self.uts]
        for em, ut in zip(ems, uts):
            ut.EMOTION = em
        pairs = list()
        cause, effect = graph.nonzero(as_tuple=True)
        for c, e in zip(cause.tolist(), effect.tolist()):
            span = spans[c, e, :len(uts[c])].detach().clone()
            if span.sum() < 1:
                continue 
            pairs.append(CauseRelation(uts[c], uts[e], span))
        return Conversation(self.id, uts, pairs, 1)
    
    def update2(self, graph: torch.Tensor, ems: List[str]):
        graph = graph.cpu()
        uts = [ut.copy() for ut in self.uts]
        for em, ut in zip(ems, uts):
            ut.EMOTION = em
        pairs = []
        cause, effect = graph.nonzero(as_tuple=True)
        for c, e in zip(cause.tolist(), effect.tolist()):
            pairs.append(CauseRelation(uts[c], uts[e], None))
        return Conversation(self.id, uts, pairs, 2)
    
    
    @property 
    def weights(self) -> Dict[str, float]:
        weights = dict()
        for pair in self.pairs:
            try:
                weights[pair.EMOTION] += 1
            except:
                weights[pair.EMOTION] = 1
        return {em: weights[em]/sum(weights.values()) for em in weights.keys()}

            
