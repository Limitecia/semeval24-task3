from __future__ import annotations
from data.utterance import Utterance
from typing import List, Union, Optional 
import torch, string
import numpy as np 

class CauseRelation:

    def __init__(self, CAUSE: Utterance, EFFECT: Utterance,  SPAN: Optional[Union[torch.Tensor, str]]):
        self.CAUSE = CAUSE
        self.EFFECT = EFFECT
        self.EMOTION = EFFECT.EMOTION
        if isinstance(SPAN, str):
            SPAN = CAUSE.get_mask(SPAN)
        self.SPAN = SPAN


    @property
    def span_len(self):
        start, end = self.SPAN.nonzero().flatten().tolist()
        return (end-start)+1

    def __repr__(self):
        return f'{self.CAUSE.ID} -({self.EFFECT.EMOTION})-> {self.EFFECT.ID}'
    
    def to_list(self, span_text: bool) -> List[str, str]:
        r"""
        Returns the original format of the causal relations.
        
        Arguments:
            span_text (bool): Whether to use the span text or the span positions.
        """
        if self.SPAN is not None:
            indices = self.SPAN.nonzero().flatten().tolist()
            span = ' '.join(np.array(self.CAUSE.TEXT.split())[indices])
            if span_text:
                return [f'{self.EFFECT.ID}_{self.EMOTION}', f'{self.CAUSE.ID}_{span}']
            else:
                # check if the span has punctuation 
                span_words = span.split()
                while len(span_words) > 0 and span_words[0] in string.punctuation:
                    indices.pop(0)
                    span_words.pop(0)
                while len(span_words) > 0 and span_words[-1] in string.punctuation:
                    indices.pop(-1)
                    span_words.pop(-1)
                if len(span_words) == 0:
                    return None
                start, end = indices[0], indices[-1]+1
                return [f'{self.EFFECT.ID}_{self.EMOTION}', f'{self.CAUSE.ID}_{start}_{end}']
        else: # subtask 2
            return [f'{self.EFFECT.ID}_{self.EMOTION}',  f'{self.CAUSE.ID}']

    
    def eq(self, other: CauseRelation, labeled: bool = True) -> bool:
        eq = (other.CAUSE.ID == self.CAUSE.ID) and (other.EFFECT.ID == self.EFFECT.ID)
        if labeled:
            eq = eq and (other.CAUSE.EMOTION == self.CAUSE.EMOTION)
        return eq 
    

        
        

