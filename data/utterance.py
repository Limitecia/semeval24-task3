from __future__ import annotations
import re, torch 
import numpy as np 
from typing import Optional , Dict


class Utterance:
    FIELDS = ['ID', 'TEXT', 'SPEAKER', 'EMOTION', 'FRAME', 'AUDIO']

    def __init__(self, ID: int, TEXT: str, SPEAKER: str, EMOTION: str, FRAME: Optional[str] = None, AUDIO: Optional[str] = None):
        self.ID = ID
        self.TEXT = TEXT.strip()
        self.SPEAKER = SPEAKER
        self.EMOTION = EMOTION
        self.FRAME = FRAME
        self.AUDIO = AUDIO

    def get_mask(self, span: str) -> torch.Tensor:
        tokens = span.strip().split()
        span = ' '.join(tokens)
        assert span in self.TEXT 
        start = None
        n = len(span.split())
        words = self.TEXT.split()
        for i in range(len(words)):
            if ' '.join(words[i:(i+n)]) == span:
                start = i
        mask = torch.zeros(len(words), dtype=torch.bool)
        mask[start:(start+n)] = True
        assert (np.array(words)[mask.numpy()] == np.array(span.split())).all(), 'span is incorrect'
        return mask 

    def __repr__(self):
        return f'Utterance(ID={self.ID}, text={self.TEXT}, speaker={self.SPEAKER}, emotion={self.EMOTION})'

        
    def copy(self) -> Utterance:
        return Utterance(self.ID, self.TEXT, self.SPEAKER, self.EMOTION, self.FRAME, self.AUDIO)

    def __len__(self):
        return len(self.TEXT.split())

    def to_dict(self) -> Dict[str, str]:
        return {
            'utterance_ID': self.ID, 
            'text': self.TEXT, 
            'speaker': self.SPEAKER,
            'emotion': self.EMOTION,
            'video_name': self.FRAME.split('/')[-1] if self.FRAME is not None else ''
        }
    
    
    @classmethod
    def from_dict(cls, data: dict, folder: Optional[str] = None) -> Utterance:
        if 'emotion' not in data.keys():
            data['emotion'] = ''
        if folder is not None:
            data['video'] = folder + '/video/' + data['video_name']
            data['audio'] = folder + '/audio/' + data['video_name'].replace('.mp4', '.wav')
        else:
            data['video'], data['audio'] = None, None
        return Utterance(data['utterance_ID'], data['text'], data['speaker'], data['emotion'], data['video'], data['audio'])