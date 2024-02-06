from __future__ import annotations
from data.relation import CauseRelation
from data.utterance import Utterance
from typing import List, Dict, Set
import torch
import numpy as np

class Conversation:
    NO_CAUSE = '_'
    SEP = '_'

    def __init__(self, id: int, uts: List[Utterance], pairs: List[CauseRelation]):
        self.id = id
        self.uts = uts
        self.pairs = pairs
        self.GRAPH = None
        self.SPAN = None
        self.build()


    def build(self):
        # self.GRAPH = np.repeat(Conversation.NO_CAUSE, len(self)**2).reshape(len(self), len(self)).astype(np.object_)
        self.GRAPH = torch.zeros(len(self), len(self), dtype=torch.bool)
        self.LGRAPH = torch.zeros(len(self), len(self), dtype=torch.int32)
        self.SPAN = torch.zeros(len(self), len(self), max(map(len, self.uts))+1)
        rem = []
        for i, pair in enumerate(self.pairs):
            if self.GRAPH[pair.CAUSE.ID - 1, pair.EFFECT.ID - 1]:
                rem.append(i)
                continue 
            self.GRAPH[pair.CAUSE.ID - 1, pair.EFFECT.ID - 1] = True
            self.SPAN[pair.CAUSE.ID - 1, pair.EFFECT.ID -1][:len(pair.SPAN)] = pair.SPAN.clone()
        self.pairs = [pair for i, pair in enumerate(self.pairs) if i not in rem]
        for ut in self.uts:
            self.SPAN[ut.ID - 1, :, (len(ut)+1):] = -1
        for field in Utterance.FIELDS:
            self.__setattr__(field, [getattr(ut, field) for ut in self.uts])
            
    @classmethod
    def from_dict(cls, data: dict) -> Conversation:
        id = data.pop('conversation_ID')
        conversation = data.pop('conversation')

        # extract uts
        uts = [Utterance.from_dict(ut) for ut in conversation]

        # extract cause-relation pairs
        if 'emotion-cause_pairs' not in data.keys():
            pairs = []
        else:
            pairs = data.pop('emotion-cause_pairs')
            pairs = [(*pair[0].split(Conversation.SEP), *pair[1].split(Conversation.SEP)) for pair in pairs]
            pairs = [CauseRelation(CAUSE=uts[int(c) - 1], EFFECT=uts[int(e) - 1], SPAN=span) for e, _, c, span in pairs]
        return Conversation(id, uts, pairs)

    def __len__(self):
        return len(self.uts)
    
    def to_dict(self, submission: bool = True):
        return {'conversation_ID': self.id, 'conversation': [ut.to_dict() for ut in self.uts], 
                'emotion-cause_pairs': [rel.to_list(text=not submission) for rel in self.pairs]}

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


    def update(self, graph: torch.Tensor, ems: List[str], spans: torch.Tensor):
        graph, spans = graph.cpu(), spans.cpu()
        uts = [ut.copy() for ut in self.uts]
        for em, ut in zip(ems, uts):
            ut.EMOTION = em
        pairs = list()
        cause, effect = graph.nonzero(as_tuple=True)
        for c, e in zip(cause.tolist(), effect.tolist()):
            span = spans[c, e, :(len(uts[c]) + 1)].detach().clone()
            span = span & uts[c].punct_mask
            if span.sum() < 2:
                continue 
            pairs.append(CauseRelation(uts[c], uts[e], span))
        return Conversation(self.id, uts, pairs)
    
    
    @property 
    def weights(self) -> Dict[str, float]:
        weights = dict()
        for pair in self.pairs:
            try:
                weights[pair.EMOTION] += 1
            except:
                weights[pair.EMOTION] = 1
        return {em: weights[em]/sum(weights.values()) for em in weights.keys()}

            
    def eval(self, other: Conversation) -> Dict[str, float]:
        weights = self.weights 
        ngold, npred = len(self.pairs), len(other.pairs)
        tp = 0 
        for gold in self.pairs:
            for pred in other.pairs:
                if gold.eq(pred, labeled=False):
                    tp += 1 
                    break 
        egold, epred = self.emotions, other.emotions
        etp = {em: [] for em in egold.keys()}
        for gold in self.pairs:
            for pred in other.pairs:
                if gold.eq(pred, labeled=True):
                    etp[gold.EMOTION].append((gold, pred))
                    break 
        ER = {em: (len(etp[em])/egold[em]) for em in egold.keys()}
        EP = {em: (len(etp[em])/epred[em]) if em in epred.keys() else 0 for em in egold.keys()}
        result = {
            'UR': tp/ngold if ngold > 0 else tp == ngold, 'UP': tp/npred if npred > 0 else 0,
            'wER': sum(ER[em]*weights[em] for em in egold.keys()),
            'wEP': sum(EP[em]*weights[em] for em in egold.keys()),
            'wEF': sum(((2*ER[em]*EP[em])/(ER[em]+EP[em]))*weights[em] if (ER[em]+EP[em] > 0) else 0 for em in egold.keys()),
        }
        result['UF'] = (2*result['UR']*result['UP'])/(result['UR']+result['UP']) if (result['UR']+result['UP']) > 0 else 0
        OR, OP = dict(), dict()
        for em in egold.keys():
            stp = sum(gold.overlap(pred) for gold, pred in etp[em])
            sgold = sum(sum(gold.SPAN) for gold in self.pairs if gold.EMOTION == em)
            spred = sum(sum(pred.SPAN) for pred in other.pairs if pred.EMOTION == em)
            OR[em] = stp/sgold if sgold > 0 else int(stp == sgold)
            OP[em] = stp/spred if spred > 0 else int(spred == sgold)
        result['wOR'] = sum(OR[em]*weights[em] for em in egold.keys())
        result['wOP'] = sum(OP[em]*weights[em] for em in egold.keys())
        result['wOF'] = sum(((2*OR[em]*OP[em])/(OR[em]+OP[em]))*weights[em] if OR[em]+OP[em] > 0 else 0for em in egold.keys())
        return result