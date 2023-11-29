from __future__ import annotations
from typing import Dict, Union, Iterable, Optional, Set, List
import torch, pickle, os
from transformers import AutoTokenizer
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable
from tqdm import tqdm
import numpy as np
from utils.fns import flatten_list

UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'


class Tokenizer:
    """
    Implementation of a simple Tokenizer.
    """
    EXTENSION = 'tkz'
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
        """
        Creates an instance of a Tokenizer.
        :param field: Field name associated with the tokenizer.
        :param max_words: Maximum number of words stored. If None, all words introduced in the tokenizer are stored.
        :param preprocess: Function to preprocess tokens before storing. If None, no preprocessing is applied.
        """
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

    def preprocess(self, token: str):
        """
        Preprocess new input token.
        """
        return token.lower() if self.lower else token

    def _encode(self, token: str) -> int:
        """
        Returns corresponding index from a non-preprocessed token.
        """
        token = self.preprocess(token)
        try:
            return self.vocab[token]
        except KeyError:
            return self.vocab[self.unk_token]

    def _decode(self, index: int) -> str:
        """
        Recovers original token from index.
        """
        return self.inv_vocab[index]

    def __getitem__(self, item: Union[int, str]):
        """
        Computes tokenization / detokenization.
        :param item: Input (token or index).
        :returns: Index (if the input is a token) or token (if the input is a string).
        """
        if isinstance(item, str):
            return self._encode(item)
        elif isinstance(item, int):
            return self._decode(item)

    def encode(self, tokens: Iterable[str]) -> torch.Tensor:
        indices = [self._encode(token) for token in tokens]
        if self.bos_token:
            indices = [self.bos_index] + indices
        if self.eos_token:
            indices.append(self.eos_index)
        return torch.tensor(indices)

    def decode(self, indices: Union[Iterable[int], torch.Tensor], remove_unk: bool = False):
        """
        Computes detokenization for multiple indices.
        """
        if isinstance(indices, torch.Tensor):
            if len(indices.shape) > 1:
                return [self.decode(x) for x in indices.tolist()]
            indices = indices.tolist()
        return [self._decode(index) for index in indices if (not remove_unk or index != self.unk_index)]

    def add(self, token: str):
        """
        Adds a new non-preprocessed token to the vocabulary.
        """
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


    def fit(self, tokens: Iterable[str], show: bool = False):
        """
        Fits tokenizer with an iterable of non-preprocessed tokens. First it updates the counter dictionary.
        If a maximum number of words has been fixed, only the top `max_words` occurrences are included in the vocabulary.
        """
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

    def __len__(self) -> int:
        return len(self.vocab)

    @property
    def weights(self) -> torch.Tensor:
        """
        Computes weights for each token of the vocabulary based on its occurrences.
        weight =
            1 - freq/total_freq                 ~ if token is not special
            1 - (total_freq-1)/total_freq       ~ if token is special
        """
        weights = list()
        total = sum(self.counter.values())
        for i in range(len(self)):
            token = self.inv_vocab[i]
            if token in self.counter.keys():
                freq = self.counter[token]
                weight = 1-freq/total
            else:
                weight = 1-(total-1)/total
            weights.append(weight)
        return torch.tensor(weights)

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


    def save(self, path: str):
        with open(path, 'wb') as writer:
            pickle.dump(self.__dict__, writer)

    @classmethod
    def load(cls, path: str) -> Tokenizer:
        extension = path.split('.')[-1]
        if extension == PretrainedTokenizer.EXTENSION:
            return PretrainedTokenizer.load(path)
        elif extension == StaticTokenizer.EXTENSION:
            return StaticTokenizer.load(path)
        elif extension == NullTokenizer.EXTENSION:
            return NullTokenizer.load(path)
        with open(path, 'rb') as reader:
            data = pickle.load(reader)

        tokenizer = cls(field=data['field'], max_words=data['max_words'], lower=data['lower'],
                        pad_token=data['pad_token'], unk_token=data['unk_token'], bos_token=data['bos_token'], eos_token=data['eos_token'])
        tokenizer.vocab = data['vocab']
        tokenizer.inv_vocab = data['inv_vocab']
        tokenizer.counter = data['counter']
        tokenizer.specials = data['specials']
        return tokenizer




class PretrainedTokenizer(Tokenizer):
    EXTENSION = 'tkz-pre'
    TRAINABLE = True
    def __init__(
        self,
        field: str,
        name: str,
        lower: bool,
        fix_len: int,
        bos: bool = False,
        eos: bool = False
    ):
        self.field = field
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name, padding_side='right')
        self.fix_len = fix_len
        self.lower = lower
        self.counter = dict()
        self.bos = bos
        self.eos = eos

        self.bos_token = self.tokenizer.bos_token or self.tokenizer.pad_token
        self.eos_token = self.tokenizer.eos_token or self.tokenizer.pad_token

    def __repr__(self):
        return f'PretrainedTokenizer(field={self.field}, name={self.name}, fix_len={self.fix_len})'

    @property
    def pad_index(self):
        if self.tokenizer.pad_token_id is not None:
            return self.tokenizer.pad_token_id
        else:
            return self.tokenizer.eos_token_id

    @property
    def bos_index(self):
        return self.tokenizer.bos_token_id

    def encode(self, tokens: Iterable[str]) -> torch.Tensor:
        tokens = list(tokens)
        if self.bos:
            tokens = [self.bos_token] + tokens
        if self.eos:
            tokens += [self.eos_token]
        tokens = list(map(self.preprocess, tokens))
        ids = self.tokenizer(tokens, padding='max_length', truncation=True, add_special_tokens=False, max_length=self.fix_len, return_tensors='pt')['input_ids']
        return ids

    def __len__(self):
        return self.tokenizer.vocab_size

    def fit(self, tokens: Iterable[str], show: bool = True):
        for token in tqdm(tokens, desc=f'{self.field} pretrained tokenizer counting', disable=not show):
            token = self.preprocess(token)
            try:
                self.counter[token] += 1
            except KeyError:
                self.counter[token] = 1

    def save(self, path: str):
        objects = dict(field=self.field, name=self.name, fix_len=self.fix_len, lower=self.lower, counter=self.counter)
        with open(path, 'wb') as writer:
            pickle.dump(objects, writer)


    @classmethod
    def load(cls, path: str) -> PretrainedTokenizer:
        with open(path, 'rb') as reader:
            data = pickle.load(reader)

        counter = data.pop('counter')
        tokenizer = cls(**data)
        tokenizer.counter = counter
        return tokenizer

class NullTokenizer(Tokenizer):
    EXTENSION = 'tkz-null'
    TRAINABLE = False
    def __init__(self, field: str):
        super().__init__(field, max_words=None, lower=False, pad_token=None, unk_token=None)

    def __repr__(self):
        return f'NullTokenizer(field={self.field})'

    @property
    def pad_index(self):
        return -1

    def encode(self, tokens: Iterable[int]) -> torch.Tensor:
        return torch.tensor(tokens)

    def fit(self, tokens: Iterable[str], show: bool = False):
        pass

    @classmethod
    def load(cls, path: str) -> NullTokenizer:
        with open(path, 'rb') as reader:
            data = pickle.load(reader)
        return NullTokenizer(field=data['field'])




class StaticTokenizer(Tokenizer):
    EXTENSION = 'tkz-static'
    TRAINABLE = False

    def __init__(self, field: str, vocab: List[str], pad_token: str = PAD_TOKEN):
        super().__init__(field, max_words=None, lower=False, pad_token=pad_token)
        self.original = vocab
        for word in vocab:
            self.add(word)

    def __repr__(self):
        return f'StaticTokenizer(field={self.field}, specials={self.specials})'

    def encode(self, tokens: List[str]) -> torch.Tensor:
        return torch.tensor([self._encode(token) for token in tokens])

    def fit(self, tokens: Iterable[str], show: bool = False):
        pass

    @classmethod
    def load(cls, path: str) -> StaticTokenizer:
        with open(path, 'rb') as reader:
            data = pickle.load(reader)

        return StaticTokenizer(field=data['field'], vocab=data['original'])

class MatrixTokenizer(StaticTokenizer):
    EXTENSION = 'tkz-mat'
    TRAINABLE = False
    def __init__(self, field: str, vocab: List[str], pad_token: str):
        super().__init__(field, vocab, pad_token)

    def __repr__(self):
        return f'MatrixTokenizer(field={self.field}, pad_token={self.pad_token})'

    def encode(self, tokens: np.array) -> torch.Tensor:
        return torch.tensor([list(map(self.vocab.get, row)) for row in tokens.tolist()])






def fit(tokenizer: Tokenizer, data: List, show: bool) -> Tokenizer:
    if isinstance(tokenizer, NullTokenizer):
        return tokenizer
    elif isinstance(tokenizer, MatrixTokenizer):
        tokenizer.fit([getattr(item, tokenizer.field) for item in data], show)
    else:
        tokenizer.fit(flatten_list([getattr(item, tokenizer.field) for item in data]), show)

    return tokenizer

def parallel(tokenizers: List[Tokenizer], data: list, num_workers: int = os.cpu_count(), show: bool = True):
    fields = [t.field for t in tokenizers]
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = list()
        for i, t in enumerate(tokenizers):
            if t.TRAINABLE:
                futures.append(pool.submit(fit, t, data, show))
        for f in range(len(futures)):
            t = futures.pop(0).result()
            tokenizers[fields.index(t.field)] = t
    return tokenizers



