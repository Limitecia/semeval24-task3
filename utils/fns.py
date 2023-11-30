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


def multipad(inputs: List[torch.Tensor], target_dims: Tuple[int], pad_value: int, dtype = torch.float32):
    for i, target_dim in enumerate(target_dims[::-1]):
        def mask(input: torch.Tensor) -> torch.Tensor:
            dims = list(iter(input.shape))
            dims[-(i+1)] = target_dim-dims[-(i+1)]
            return torch.full(dims, fill_value=pad_value, dtype=dtype)

        inputs = [
            torch.concat([input, mask(input)], dim=-(i+1))
            for input in inputs
        ]
    return torch.stack(inputs, dim=0)

