from __future__ import annotations
from data.relation import CauseRelation
from data.utterance import Utterance
from typing import List, Dict
import torch

class Conversation:
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
        self.EMOTIONS = None
        self.SPAN = None
        self.build()


    def build(self):
        self.GRAPH = torch.zeros(len(self.utterances), len(self.utterances))
        self.EMOTIONS = []
        self.SPAN = []
        for pair in self.pairs:
            self.GRAPH[pair.EFFECT.ID - 1, pair.CAUSE.ID - 1] = 1
            self.EMOTIONS.append(pair.EMOTION)
            self.SPAN.append(pair.SPAN)
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

