import torch
from typing import List, Tuple, Union
from transformers.feature_extraction_utils import BatchFeature

def flatten_list(lists):
    result = list()
    for item in lists:
        result += item
    return result

def flatten_set(sets):
    result = set()
    for item in sets:
        result |= set(item)
    return result


def to(device, *tensors):
    result = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) or isinstance(tensor, BatchFeature):
            result.append(tensor.to(device))
        else:
            result.append(to(device, *tensor))
    return result

def cuda(tensors):
    result = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            result.append(tensor.cuda())
        else:
            result.append(cuda(tensor))
    return result


def onehot(indices: List[torch.Tensor], num_classes: int) -> torch.Tensor:
    mask = torch.zeros((len(indices), num_classes))
    for i, cols in enumerate(indices):
        mask[i, cols.tolist()] = 1
    return mask.to(torch.bool)

def split(items: list, lens: Union[List[int], int]):
    result = []
    items = list(items)
    if isinstance(lens, int):
        while len(items) > 0:
            result.append([items.pop(0) for _ in range(min(len(items), lens))])
    else:
        for l in lens:
            result.append([items.pop(0) for _ in range(l)])
    return result


def pad(x: torch.Tensor, pad_value: int, target_dims: Tuple[int], dims: Tuple[int]):
    for target_dim, dim in zip(target_dims, dims):
        s = list(x.shape)
        s[dim] = target_dim - s[dim]
        padding = torch.zeros(s).fill_(pad_value)
        x = torch.concat([x, padding], dim=dim)
    return x


def expand_mask(x: torch.Tensor) -> torch.Tensor:
    mask = x.unsqueeze(-1).expand(*[-1 for _ in x.shape], x.shape[-1]).clone()
    dim0, dim1 = (~x).nonzero().T.tolist()
    mask[dim0, :, dim1] = False 
    return mask
    
    
    
def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min().item())/(x.max().item() - x.min().item())