from __future__ import annotations
import re, torch 
from typing import List


class Utterance:
    FIELDS = ['ID', 'TEXT', 'SPEAKER', 'EMOTION']

    def __init__(self, ID: int, TEXT: str, SPEAKER: str, EMOTION: str):
        self.ID = ID
        self.TEXT = TEXT.strip()
        self.SPEAKER = SPEAKER
        self.EMOTION = EMOTION

    def get_mask(self, span: str) -> torch.Tensor:
        tokens = span.strip().split()
        if tokens[0] in ['.', '...']:
            tokens.pop(0)
        if tokens[-1] in ['.', '...']:
            tokens.pop(-1)
        span = ' '.join(tokens)
        assert span in self.TEXT 
        start = None
        n = len(span.split())
        words = self.TEXT.split()
        for i in range(len(words)):
            if ' '.join(words[i:(i+n)]) == span:
                start = i
        mask = torch.zeros(len(words) + 1, dtype=torch.bool)
        mask[start:(start+n+1)] = True
        assert len(mask) == (len(self) + 1), 'span mask is not correct'
        assert mask.sum() == len(span.split()) + 1, 'span mask is not correct'
        return mask 

    def __repr__(self):
        return f'Utterance(ID={self.ID}, text={self.TEXT}, speaker={self.SPEAKER}, emotion={self.EMOTION})'

    @classmethod
    def from_dict(cls, data: dict) -> Utterance:
        if 'emotion' not in data.keys():
            data['emotion'] = ''
        return Utterance(data['utterance_ID'], data['text'], data['speaker'], data['emotion'])
        
    def copy(self) -> Utterance:
        return Utterance(self.ID, self.TEXT, self.SPEAKER, self.EMOTION)

    def __len__(self):
        return len(self.TEXT.split())

    def to_dict(self) -> dict:
        return {
            'utterance_ID': self.ID, 
            'text': self.TEXT, 
            'speaker': self.SPEAKER,
            'emotion': self.EMOTION
        }

    @property 
    def punct_mask(self) -> torch.Tensor:
        words = self.TEXT.split()
        mask = torch.ones(len(words) + 1, dtype=torch.bool)
        if words[0] in ['.', '...']:
            mask[0] = False 
        if words[-1] in ['.', '...']:
            mask[-1] = False
        return mask 