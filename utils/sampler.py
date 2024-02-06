from torch.utils.data import Sampler
from typing import List, Iterator
import torch
from random import shuffle

class LengthSampler(Sampler):
    def __init__(self, lens: torch.Tensor, batch_size: int, shuffle: bool = True):
        self.lens = lens 
        self.indices = torch.argsort(self.lens, descending=True)
        self.batch_size = batch_size
        
        self.n = 0
        self.batches = [0]
        count = 0
        for i, l in enumerate(torch.sort(lens, descending=True).values):
            if count > batch_size:
                self.n += 1
                count = 0
                self.batches.append(i) 
            else:
                count += l 
        self.batches.append(len(lens))
        self.n += 1
        self._shuffle = shuffle
        
    def __len__(self):
        return self.n
    
    def shuffle(self):
        lens = torch.sort(self.lens, descending=True).values
        for l in lens.unique():
            indices = self.indices[lens == l].tolist()
            shuffle(indices)
            self.indices[lens == l] = torch.tensor(indices)
    
    def __iter__(self):
        if self._shuffle:
            self.shuffle()
        for i in range(len(self.batches) - 1):
            yield self.indices[self.batches[i]:self.batches[i+1]]
        
            
        
        
        
        
        