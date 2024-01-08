from __future__ import annotations
from typing import Dict, Union, Iterable, Optional, Set, List
import torch, pickle, os
from transformers import AutoTokenizer
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Callable
from tqdm import tqdm
import numpy as np
from utils.fns import flatten_list, multipad
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, PackedSequence


UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'


class Tokenizer:
    """
    Implementation of a simple Tokenizer.
    """
    EXTENSION = 'tkz'
    TRAINABLE = True
    PAD = True

    def __init__(
        self,
        field: str,
        max_words: Optional[int] = None,
        lower: bool = False,
        pad_token: Optional[str] = PAD_TOKEN,
        unk_token: Optional[str] = UNK_TOKEN,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        fn: Optional[Callable] = None
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
        self.fn = fn

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

    def preprocess(self, token: str) -> str:
        if self.lower:
            token = token.lower()
        if self.fn:
            token = self.fn(token)
        return token

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
        return torch.tensor(indices, dtype=torch.int32)

    def encode_batch(self, batch: List[Iterable[str]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        unpadded = [self.encode(tokens) for tokens in batch]
        if self.PAD:
            try:
                return pad_sequence(unpadded, batch_first=True, padding_value=self.pad_index)
            except:
                print(unpadded)
        return unpadded

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

    # @classmethod
    # def load(cls, path: str) -> Tokenizer:
    #     extension = path.split('.')[-1]
    #     if extension == PretrainedTokenizer.EXTENSION:
    #         return PretrainedTokenizer.load(path)
    #     elif extension == StaticTokenizer.EXTENSION:
    #         return StaticTokenizer.load(path)
    #     elif extension == NullTokenizer.EXTENSION:
    #         return NullTokenizer.load(path)
    #     with open(path, 'rb') as reader:
    #         data = pickle.load(reader)
    #
    #     tokenizer = cls(field=data['field'], max_words=data['max_words'], lower=data['lower'],
    #                     pad_token=data['pad_token'], unk_token=data['unk_token'], bos_token=data['bos_token'], eos_token=data['eos_token'])
    #     tokenizer.vocab = data['vocab']
    #     tokenizer.inv_vocab = data['inv_vocab']
    #     tokenizer.counter = data['counter']
    #     tokenizer.specials = data['specials']
    #     return tokenizer




class WordTokenizer(Tokenizer):
    EXTENSION = 'tkz-word'
    TRAINABLE = False

    def __init__(
        self,
        field: str,
        name: str,
        lower: bool,
        bos: bool = False,
        eos: bool = False,
        fix_len: int = 5
    ):
        self.field = field
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name, padding_side='right')
        self.lower = lower
        self.bos = bos
        self.eos = eos
        self.fn = None
        self.fix_len = fix_len

        self.tokenizer.eos_token_id = self.tokenizer.pad_token_id

        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.pad_token = self.tokenizer.pad_token
        self.unk_token = self.tokenizer.unk_token

    @property
    def pad_index(self):
        return self.tokenizer.pad_token_id

    @property
    def bos_index(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_index(self):
        return self.tokenizer.eos_token_id

    @property
    def unk_index(self):
        return self.tokenizer.unk_token_id

    def __repr__(self):
        return f'WordTokenizer(field={self.field}, name={self.name}, bos={self.bos}, eos={self.eos})'

    def preprocess(self, words: List[str]) -> List[str]:
        if self.lower:
            words = [word.lower() for word in words]
        if self.bos:
            words = [self.bos_token] + words
        if self.eos:
            words.append(self.eos_token)
        return words

    def encode(self, texts: Iterable[str]) -> torch.Tensor:
        r"""
        Receives texts of a conversation and returns a PaddedSequence.
        Args:
            texts (Iterable[str]): ``[seq_len] ~ conv_len``.

        Returns:
            ~torch.Tensor: ``[conv_len, pad(seq_len), fix_len]``
        :return:
        """
        words = [self.preprocess(text.split()) for text in texts]
        lens = list(map(len, words))
        tokens = self.tokenizer(flatten_list(words), padding='max_length', truncation=True, max_length=self.fix_len, add_special_tokens=False, return_tensors='pt')
        ids = pad_sequence(tokens['input_ids'].split(lens), batch_first=True, padding_value=self.pad_index)
        return ids

    def encode_batch(self, batch: List[Iterable[str]]) -> torch.Tensor:
        r"""
        Receives texts from a batch of conversations.
        Args:
            batch (List[Iterable[str]]): ``[[seq_len] ~ conv_len] ~ batch_size
        Returns:
            ~ torch.Tensor: ``[batch_size, pad(conv_len), pad(seq_len), fix_len]``
        """
        conv_lens = list(map(len, batch))
        unpadded = self.encode(flatten_list(batch)).split(conv_lens, dim=0)
        # pad sequence lengths
        padded = pad_sequence(unpadded, batch_first=True, padding_value=self.pad_index)
        return padded

    def __len__(self):
        return self.tokenizer.vocab_size

    def fit(self, tokens: Iterable[str], show: bool = True):
        pass

    def save(self, path: str):
        objects = dict(field=self.field, name=self.name, lower=self.lower, bos=self.bos, eos=self.eos)
        with open(path, 'wb') as writer:
            pickle.dump(objects, writer)


    @classmethod
    def load(cls, path: str) -> WordTokenizer:
        with open(path, 'rb') as reader:
            data = pickle.load(reader)

        counter = data.pop('counter')
        tokenizer = cls(**data)
        tokenizer.counter = counter
        return tokenizer



class LabeledGraphTokenizer(Tokenizer):
    r"""
    Implementation of a graph tokenizer. It does not store a vocabulary, but streams a graph per instance.
    """

    EXTENSION = 'tkz-lgraph'
    TRAINABLE = False
    PAD = True

    def __init__(self, field: str, vocab: Set[str], pad_token: str):
        super().__init__(field, max_words=None, lower=False, pad_token=pad_token, unk_token=None)
        for word in vocab:
            self.add(word)

    def __repr__(self):
        return f'GraphTokenizer(field={self.field})'

    def encode(self, graph: np.array) -> torch.Tensor:
        return torch.tensor([[self.vocab[t] for t in row] for row in graph.tolist()])

    def encode_batch(self, batch: List[torch.Tensor]) -> torch.Tensor:
        r"""
        Batch padding.
        Args:
            batch (List[torch.Tensor]): ``[seq_len, seq_len] ~ batch_size``
        Returns:
            ~ torch.Tensor: ``[batch_size, pad(seq_len), pad(seq_len)]``
        """
        lens = list(map(lambda x: x.shape[1], batch))
        padded1 = pad_sequence(flatten_list(self.encode(x).unbind(-1) for x in batch), batch_first=True, padding_value=self.pad_index).split(lens)
        padded0 = pad_sequence(padded1, batch_first=True, padding_value=self.pad_index)
        return padded0.to(torch.int32)

    def fit(self, tokens: Iterable[str], show: bool = False):
        pass

    @classmethod
    def load(cls, path: str) -> GraphTokenizer:
        with open(path, 'rb') as reader:
            data = pickle.load(reader)
        return GraphTokenizer(field=data['field'])



class GraphTokenizer(LabeledGraphTokenizer):
    r"""
    Implementation of a graph tokenizer. It does not store a vocabulary, but streams a graph per instance.
    """

    EXTENSION = 'tkz-graph'
    TRAINABLE = False
    PAD = True

    def __init__(self, field: str):
        super().__init__(field, vocab=set(), pad_token=None)

    def __repr__(self):
        return f'GraphTokenizer(field={self.field})'

    @property
    def pad_index(self):
        return 0

    def encode(self, graph: torch.Tensor) -> torch.Tensor:
        return graph

    def encode_batch(self, batch: List[torch.Tensor]) -> torch.Tensor:
        r"""
        Batch padding.
        Args:
            batch (List[torch.Tensor]): ``[seq_len, seq_len] ~ batch_size``
        Returns:
            ~ torch.Tensor: ``[batch_size, pad(seq_len), pad(seq_len)]``
        """
        lens = list(map(lambda x: x.shape[0], batch))
        padded1 = pad_sequence(flatten_list(self.encode(x).unbind(0) for x in batch), batch_first=True, padding_value=self.pad_index).split(lens)
        padded0 = pad_sequence(padded1, batch_first=True, padding_value=self.pad_index)
        return padded0.to(torch.bool)

    def fit(self, tokens: Iterable[str], show: bool = False):
        pass

    @classmethod
    def load(cls, path: str) -> GraphTokenizer:
        with open(path, 'rb') as reader:
            data = pickle.load(reader)
        return GraphTokenizer(field=data['field'])



class SpanTokenizer(Tokenizer):
    EXTENSION = 'tkz-sub'
    PAD = True
    TRAINABLE = False

    def __init__(
        self,
        field: str,
        max_words: Optional[int] = None,
        lower: bool = False
    ):
        super().__init__(field, max_words, lower, pad_token=None, unk_token=None, bos_token=None, eos_token=None)

    @property
    def pad_index(self):
        return -1

    def encode(self, spans: torch.Tensor) -> torch.Tensor:
        return spans

    def encode_batch(self, batch: List[torch.Tensor]) -> torch.Tensor:
        r"""
        Receives a batch of conversations and returns a padded 3D tensor of spans.
        Args:
            batch (List[torch.BoolTensor]): ``[[conv_len, conv_len, max(ut_len)] ~ batch_size``
        Returns:
            ~ torch.BoolTensor: ``[batch_size, max(conv_len), max(conv_len), max(ut_len)]``
        """
        # pad utterance length
        max_ut_len = max(map(lambda b: b.shape[-1], batch))
        max_conv_len = max(map(lambda b: b.shape[0], batch))
        return multipad(batch, target_dims=(max_conv_len, max_conv_len, max_ut_len), pad_value=0, dtype=torch.bool)




def fit(tokenizer: Tokenizer, data: List, show: bool) -> Tokenizer:
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



