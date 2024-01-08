from __future__ import annotations
import re
from typing import List
import numpy as np
class Utterance:
    FIELDS = ['ID', 'TEXT', 'SPEAKER', 'EMOTION']

    def __init__(self, ID: int, TEXT: str, SPEAKER: str, EMOTION: str):
        self.ID = ID
        self.TEXT = TEXT.strip().lower()
        self.words = np.array(self.TEXT.split())
        self.SPEAKER = SPEAKER
        self.EMOTION = EMOTION

    def get_mask(self, span: str) -> np.array:
        match = re.search(span.strip(), self.TEXT)
        if match is None:
            return np.array([True for _ in self.words])
        mask = []
        i = 0
        for word in self.words:
            mask.append(i in range(match.start(), match.end()))
            i += (len(word) + 1)
        return np.array(mask)

    def __repr__(self):
        return f'Utterance(ID={self.ID}, text={self.TEXT}, speaker={self.SPEAKER}, emotion={self.EMOTION})'

    @classmethod
    def from_dict(cls, data: dict) -> Utterance:
        return Utterance(data['utterance_ID'], data['text'], data['speaker'], data['emotion'])

    def __len__(self):
        return len(self.words)