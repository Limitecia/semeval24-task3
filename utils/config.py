from __future__ import annotations
from configparser import SectionProxy

class Config:
    REMOVE = ['kwargs', 'self']

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def __call__(self) -> dict:
        return self.__dict__

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.__setattr__(key, value)
            
    def pop(self, name):
        value = getattr(self, name)
        self.__delattr__(name)
        return value 

    @classmethod
    def from_class(cls, data) -> Config:
        data = dict(filter(lambda x: not (x[0].startswith('_') or x[0] in Config.REMOVE), data.items()))
        return Config(**data)

    @classmethod 
    def from_ini(cls, section: SectionProxy) -> Config:
        data = dict()
        for param, value in section.items():
            if value.lower() == 'true':
                data[param] = True 
            elif value.lower() == 'false':
                data[param] == False 
            elif value.isdigit():
                data[param] = int(value)
            elif value.replace('.', '').isdigit():
                data[param] = float(value)
            else:
                data[param] = value
        return Config(**data)
