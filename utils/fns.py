import torch
from typing import List

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


def multipad(input: torch.Tensor, target_dims: Tuple[int])
