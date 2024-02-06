from __future__ import annotations
from data.utterance import Utterance
from typing import List, Union
import torch 

class CauseRelation:

    def __init__(self, CAUSE: Utterance, EFFECT: Utterance,  SPAN: Union[torch.Tensor, str]):
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
    
    def to_list(self, text: bool = False) -> List[str, str]:
        start, *_, end = self.SPAN.nonzero().flatten().tolist()
        if text:
            span = ' '.join(self.CAUSE.TEXT.split()[start:end])
            return [f'{self.EFFECT.ID}_{self.EMOTION}', f'{self.CAUSE.ID}_{span}']
        else:
            return [f'{self.EFFECT.ID}_{self.EMOTION}', f'{self.CAUSE.ID}_{start}_{end}']
    
    def eq(self, other: CauseRelation, labeled: bool = True) -> bool:
        eq = (other.CAUSE.ID == self.CAUSE.ID) and (other.EFFECT.ID == self.EFFECT.ID)
        if labeled:
            eq = eq and (other.CAUSE.EMOTION == self.CAUSE.EMOTION)
        return eq 
    

        
        

