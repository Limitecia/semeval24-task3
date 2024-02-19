from __future__ import annotations
from typing import Dict, Union, Iterator, Optional, Set, List
import torch, pickle, os
from transformers import AutoTokenizer
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable
from tqdm import tqdm
import numpy as np
from utils.fns import flatten_list, pad
from torch.nn.utils.rnn import pad_sequence
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'


class Tokenizer:
    EXTENSION = 'tkz'
    OBJECTS = ['field', 'max_words', 'lower', 'pad_token', 'unk_token', 'bos_token', 'eos_token', 'counter']
    TRAINABLE = True

    def __init__(
        self,
        field: str,
        max_words: Optional[int] = None,
        lower: bool = False,
        pad_token: Optional[str] = PAD_TOKEN,
        unk_token: Optional[str] = UNK_TOKEN,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None
    ):
        self.field = field
        self.lower = lower
        self.max_words = max_words

        # initialize variables
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.specials = [token for token in (self.pad_token, self.unk_token, self.bos_token, self.eos_token) if token is not None]
        self.vocab = {token: i for i, token in enumerate(self.specials)}
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}
        self.counter = dict()
        
        
    def __repr__(self):
        return f'Tokenizer(field={self.field}, specials={self.specials}, vocab_size={len(self)})'
    
    def __len__(self) -> int:
        return len(self.vocab)

    def preprocess(self, token: str) -> str:
        if self.lower:
            token = token.lower()
        return token
    
    @property 
    def objects(self) -> dict:
        return {key: getattr(self, key) for key in self.OBJECTS}
    
    @property
    def pad_index(self):
        return self.vocab[self.pad_token]

    @property
    def bos_index(self):
        return self.vocab[self.bos_token]

    @property
    def eos_index(self):
        return self.vocab[self.eos_token]

    @property
    def unk_index(self):
        return self.vocab[self.unk_token]
    
    def add(self, token: str):
        id = len(self.vocab)
        token = self.preprocess(token)
        self.vocab[token] = id
        self.inv_vocab[id] = token
        
    def empty(self):
        self.vocab = {token: i for i, token in enumerate(self.specials)}
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}
        self.counter = dict()

    @property
    def tokens(self) -> Set[str]:
        tokens = self.vocab.keys() - set(self.specials)
        return tokens
    
    @property
    def weights(self) -> torch.Tensor:
        weights = list()
        total = max(self.counter.values())
        for i in range(len(self)):
            token = self.inv_vocab[i]
            if token in self.counter.keys():
                freq = self.counter[token]
                weight = 1/np.sqrt(freq/total)
            else:
                weight = 1/np.sqrt((total-1)/total)
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.float32)**2

    def _encode(self, token: str) -> int:
        token = self.preprocess(token)
        try:
            return self.vocab[token]
        except KeyError:
            return self.vocab[self.unk_token]

    def _decode(self, index: int) -> str:
        return self.inv_vocab[index]

    def __getitem__(self, item: Union[int, str]):
        if isinstance(item, str):
            return self._encode(item)
        elif isinstance(item, int):
            return self._decode(item)

    def encode(self, tokens: Iterator[str]) -> torch.Tensor:
        indices = [self._encode(token) for token in tokens]
        if self.bos_token:
            indices = [self.bos_index] + indices
        if self.eos_token:
            indices.append(self.eos_index)
        return torch.tensor(indices, dtype=torch.int32)

    def batch_encode(self, batch: Iterator[Iterator[str]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        unpadded = [self.encode(tokens) for tokens in batch]
        return self.pad_batch(unpadded)
        
    def pad_batch(self, encoded: List[torch.Tensor]) -> Union[torch.Tensor, List[torch.Tensor]]:
        if self.pad_token:
            return pad_sequence(encoded, batch_first=True, padding_value=self.pad_index)
        else:
            return encoded 

    def decode(self, indices: torch.Tensor, remove_unk: bool = False):
        assert len(indices.shape) == 1, 'torch Tensor must have only one dimension for decoding'
        return [self._decode(index) for index in indices.tolist() if (not remove_unk or index != self.unk_index)]
    
    def batch_decode(self, indices: torch.Tensor, remove_unk: bool = False):
        if len(indices.shape) == 1:
            return self.decode(indices, remove_unk)
        else:
            return [self.batch_decode(x, remove_unk) for x in indices.unbind(0)]

    def fit(self, tokens: Iterable[str], show: bool = False):
        self.empty()
        for token in tqdm(tokens, desc=f'{self.field} tokenizer fitting', disable=not show):
            token = self.preprocess(token)
            if token in self.specials:
                continue
            try:
                self.counter[token] += 1
            except KeyError:
                self.counter[token] = 1
        if self.max_words is None:
            for token in self.counter.keys():
                self.add(token)
        else:
            for token in sorted(self.counter.keys(), key=self.counter.get, reverse=True)[:(self.max_words)]:
                self.add(token)


    def save(self, path: str):
        path += f'.{self.EXTENSION}' if not path.endswith(f'.{self.EXTENSION}') else ''
        with open(path, 'wb') as writer:
            pickle.dump(self.objects, writer)

    def load_counter(self, counter: Dict[str, int]):
        self.empty()
        for token in sorted(counter.keys(), key=counter.get, reverse=True)[:self.max_words]:
            self.add(token)
        

    @classmethod
    def load(cls, path: str) -> Tokenizer:
        extension = path.split('.')[-1]
        if extension == TextTokenizer.EXTENSION:
            return TextTokenizer.load(path)
        elif extension == GraphTokenizer.EXTENSION:
            return GraphTokenizer.load(path)
        elif extension == SpanTokenizer.EXTENSION:
            return SpanTokenizer.load(path)
        elif extension == PositionalTokenizer.EXTENSION:
            return PositionalTokenizer.load(path)
        elif extension == RawTokenizer.EXTENSION:
            return RawTokenizer.load(path)
        
        with open(path, 'rb') as reader:
            data = pickle.load(reader)
        counter = data.pop('counter')
        tokenizer = cls(**data)
        tokenizer.load_counter(counter)
        return tokenizer


class PositionalTokenizer(Tokenizer):
    EXTENSION = 'tkz-pos'    
    OBJECTS = ['field', 'max_position', 'lower', 'pad_token', 'unk_token']
    TRAINABLE = False
    
    def __init__(
        self,
        field: str,
        max_position: int,
        lower: bool = False,
        pad_token: Optional[str] = PAD_TOKEN,
        unk_token: Optional[str] = UNK_TOKEN,
        **kwargs
    ):
        self.field = field
        self.lower = lower
        self.max_position = max_position

        # initialize variables
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = None
        self.eos_token = None
        self.specials = [token for token in (self.pad_token, self.unk_token, self.bos_token, self.eos_token) if token is not None]
        
    @property 
    def pad_index(self) -> int:
        return 0 
    
    @property 
    def unk_index(self) -> int:
        return 1
    
    def __repr__(self):
        return f'PositionalTokenizer(field={self.field}, specials={self.specials}, vocab_size={len(self)})'
    
    def __len__(self) -> int:
        return self.max_position
    
    def encode(self, tokens: List[str]) -> torch.Tensor:
        unique = dict()
        indices = []
        for token in tokens:
            token = self.preprocess(token)
            try:
                index = unique[token]
            except KeyError:
                if len(unique) + len(self.specials) == self.max_position:
                    index = self.unk_index 
                else:
                    unique[token] = len(unique) + len(self.specials)
                    index = unique[token]
            indices.append(index)
        return torch.tensor(indices, dtype=torch.int32)
    
    
    def decode(self, indices: torch.Tensor, remove_unk: bool = False):
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> Tokenizer:
        with open(path, 'rb') as reader:
            data = pickle.load(reader)
        return cls(**data)
    

class RawTokenizer(Tokenizer):
    EXTENSION = 'tkz-raw'
    OBJECTS = ['field']
    TRAINABLE = False
    
    def __init__(self, field: str):
        self.field = field 
        
    def batch_encode(self, batch: List[List[str]]):
        return batch
    
    @classmethod
    def load(cls, path: str) -> RawTokenizer:
        with open(path, 'rb') as reader:
            data = pickle.load(reader)
        return cls(**data)
    

class TextTokenizer(Tokenizer):
    EXTENSION = 'tkz-text'
    OBJECTS = ['field', 'name', 'lower', 'bos', 'eos', 'fix_len']
    TRAINABLE = False


    def __init__(
        self,
        field: str,
        name: str,
        lower: bool,
        bos: bool = False,
        eos: bool = False,
        fix_len: int = 3
    ):
        self.field = field
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name, padding_side='right', do_lower_case=lower, low_memory=False)
        self.lower = lower
        self.bos = bos
        self.eos = eos
        self.fix_len = fix_len
        
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.cls_token 
        self.bos_token = self.tokenizer.bos_token
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = self.tokenizer.pad_token
        self.eos_token = self.tokenizer.eos_token 
        self.pad_token = self.tokenizer.pad_token 
        self.unk_token = self.tokenizer.unk_token 
        
        if self.tokenizer.bos_token_id is None:
            self.tokenizer.bos_token_id = self.tokenizer.cls_token_id 
            
            
    @property 
    def bos_index(self):
        return self.tokenizer.bos_token_id
        
    @property 
    def eos_index(self):
        return self.tokenizer.eos_token_id 
    
    @property 
    def pad_index(self):
        return self.tokenizer.pad_token_id 
    
    @property 
    def unk_index(self):
        return self.tokenizer.unk_token_id
        
    def __repr__(self):
        return f'TextTokenizer(field={self.field}, name={self.name}, bos={self.bos}, eos={self.eos})'
    
    def __len__(self):
        return self.tokenizer.vocab_size

    def fit(self, tokens: Iterable[str], show: bool = True):
        raise NotImplementedError
    
    def preprocess(self, words: List[str]) -> List[str]:
        if self.lower:
            words = [word.lower() for word in words]
        if self.bos:
            words = [self.bos_token] + words 
        if self.eos:
            words.append(self.eos_token)
        return words

    def encode(self, conv: List[str]) -> torch.Tensor:
        words = [self.preprocess(ut.split()) for ut in conv]
        lens = list(map(len, words))
        tokens = self.tokenizer(flatten_list(words), padding='max_length', truncation=True, max_length=self.fix_len, add_special_tokens=False, return_tensors='pt')
        ids = pad_sequence(tokens['input_ids'].split(lens), batch_first=True, padding_value=self.pad_index)
        return ids

    def batch_encode(self, batch: List[List[str]]) -> torch.Tensor:
        conv_lens = list(map(len, batch))
        unpadded = self.encode(flatten_list(batch)).split(conv_lens, dim=0)
        padded = pad_sequence(unpadded, batch_first=True, padding_value=self.pad_index)
        return padded
    
    def pad_batch(self, encoded: List[torch.Tensor]) -> torch.Tensor:
        conv_lens = [conv.shape[0] for conv in encoded]
        padded1 = pad_sequence(flatten_list(x.unbind(0) for x in encoded), True, self.pad_index)
        padded0 = pad_sequence(padded1.split(conv_lens), True, self.pad_index)
        return padded0 
        

    def decode(self, indices: torch.Tensor, remove_unk: bool = True) -> str:
        assert len(indices.shape) == 1, 'Tensor must be flatten to decode'
        return ' '.join(self.tokenizer.convert_ids_to_tokens(indices, skip_special_tokens=remove_unk))

    @classmethod
    def load(cls, path: str) -> Tokenizer:
        with open(path, 'rb') as reader:
            data = pickle.load(reader)
        return cls(**data)



class GraphTokenizer(Tokenizer):
    EXTENSION = 'tkz-graph'
    OBJECTS = ['field']
    TRAINABLE = False

    def __init__(self, field: str):
        self.field = field 

    def __repr__(self):
        return f'GraphTokenizer(field={self.field})'

    @property
    def pad_index(self):
        return 0

    def encode(self, graph: torch.Tensor) -> torch.Tensor:
        return graph

    def batch_encode(self, batch: List[torch.Tensor]) -> torch.Tensor:
        lens = list(map(lambda x: x.shape[0], batch))
        padded1 = pad_sequence(flatten_list(self.encode(x).unbind(0) for x in batch), batch_first=True, padding_value=self.pad_index).split(lens)
        padded0 = pad_sequence(padded1, batch_first=True, padding_value=self.pad_index)
        return padded0.to(torch.bool)
    
    def pad_batch(self, encoded: List[torch.Tensor]) -> torch.Tensor:
        return self.batch_encode(encoded)

    def fit(self, tokens: Iterable[str], show: bool = False):
        raise NotImplementedError
    
    @classmethod
    def load(cls, path: str) -> Tokenizer:
        with open(path, 'rb') as reader:
            data = pickle.load(reader)
        return cls(**data)



class SpanTokenizer(Tokenizer):
    EXTENSION = 'tkz-span'
    OBJECTS = ['field']
    TRAINABLE = False


    def __init__(
        self,
        field: str
    ):
        self.field = field 

    @property
    def pad_index(self):
        return -1

    def encode(self, spans: torch.Tensor) -> torch.Tensor:
        return spans

    def batch_encode(self, batch: List[torch.Tensor]) -> torch.Tensor:
        # batch: List[torch.Tensor ~ [conv_len, conv_len, pad(ut_len)]] ~ batch_size
        max_ut_len = max(map(lambda b: b.shape[-1], batch))
        max_conv_len = max(map(lambda b: b.shape[0], batch))
        
        # pad utterance length 
        batch = [pad(x, self.pad_index, (max_conv_len, max_conv_len, max_ut_len), (0, 1, 2)) for x in batch]
        return torch.stack(batch, dim=0)
    
    @classmethod
    def load(cls, path: str) -> SpanTokenizer:
        with open(path, 'rb') as reader:
            data = pickle.load(reader)
        return cls(**data)



