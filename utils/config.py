from __future__ import annotations

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

    @classmethod
    def from_class(cls, data) -> Config:
        data = dict(filter(lambda x: not (x[0].startswith('_') or x[0] in Config.REMOVE), data.items()))
        return Config(**data)
