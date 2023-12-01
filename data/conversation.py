from __future__ import annotations
from data.relation import CauseRelation
from data.utterance import Utterance
from typing import List, Dict
import torch
import numpy as np

class Conversation:
    NO_CAUSE = '_'
    SEP = '_'

    def __init__(
        self,
        id: int,
        utterances: List[Utterance],
        pairs: List[CauseRelation]
    ):
        self.id = id
        self.utterances = utterances
        self.pairs = pairs
        self.GRAPH = None
        self.SPAN = None
        self.build()


    def build(self):
        self.GRAPH = np.repeat(Conversation.NO_CAUSE, len(self)**2).reshape(len(self), len(self)).astype(np.object_)
        self.SPAN = torch.zeros(len(self), len(self), max(map(len, self.utterances)))
        for pair in self.pairs:
            self.GRAPH[pair.EFFECT.ID - 1, pair.CAUSE.ID - 1] = pair.EMOTION
            self.SPAN[pair.EFFECT.ID - 1, pair.CAUSE.ID -1][:len(pair.SPAN)] = torch.tensor(pair.SPAN)
        self.SPAN = self.SPAN.to(torch.bool)
        for field in Utterance.FIELDS:
            self.__setattr__(field, [getattr(ut, field) for ut in self.utterances])

    @classmethod
    def from_dict(cls, data: dict) -> Conversation:
        id = data.pop('conversation_ID')
        conversation = data.pop('conversation')

        # extract utterances
        utterances = [Utterance.from_dict(ut) for ut in conversation]

        # extract cause-relation pairs
        pairs = data.pop('emotion-cause_pairs')
        pairs = [(*pair[0].split(Conversation.SEP), *pair[1].split(Conversation.SEP)) for pair in pairs]
        pairs = [CauseRelation(CAUSE=utterances[int(c) - 1], EFFECT=utterances[int(e) - 1], SPAN=span) for e, _, c, span in pairs]
        return Conversation(id, utterances, pairs)

    def __len__(self):
        return len(self.utterances)

    @property
    def span_lens(self) -> List[int]:
        return [pair.span_len for pair in self.pairs]