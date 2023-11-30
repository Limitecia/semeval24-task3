from __future__ import annotations
from data.utterance import Utterance
from typing import List, Union
import numpy as np

class CauseRelation:
    def __init__(self, CAUSE: Utterance, EFFECT: Utterance,  SPAN: Union[np.array, str]):
        self.CAUSE = CAUSE
        self.EFFECT = EFFECT
        self.EMOTION = EFFECT.EMOTION
        if isinstance(SPAN, str):
            SPAN = CAUSE.get_mask(SPAN)
        self.SPAN = SPAN

    @property
    def span_len(self):
        return sum(self.SPAN)


    def __repr__(self):
        return (f'CauseRelation(\n'
                f'\tCAUSE={" ".join(self.CAUSE.words[self.SPAN])} ({self.CAUSE.ID})\n'
                f'\tEFFECT={self.EMOTION} ({self.EFFECT.ID})')


