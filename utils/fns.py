import torch
from typing import List, Tuple, Union

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


def to(tensors, device):
    result = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            result.append(tensor.to(device))
        else:
            result.append(to(tensor, device))
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

def split(items: list, lens: List[int]):
    result = []
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
    
    