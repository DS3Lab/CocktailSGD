import torch
import cupy
import numpy as np

from .utils import *

# @torch.compile()
def topr(x, ratio):
    x_flat = x.view(-1)
    numel = x_flat.numel()
    k = max(int(numel * ratio), 1)
    _, indexes = torch.topk(torch.abs(x_flat.data), k=k, sorted=False)
    masks = torch.zeros_like(x_flat, dtype=torch.uint8)
    masks[indexes] = 1
    masks = masks.view(x.shape)
    values = x.data[masks.bool()]
    return values, masks

# @torch.compile()
def topk(x, k, return_values=True, return_indices=False):
    k = max(k, 1)
    x_flat = x.view(-1)
    _, indexes = torch.topk(torch.abs(x_flat.data), k=k, sorted=False)
    masks = torch.zeros_like(x_flat, dtype=torch.uint8)
    masks[indexes] = 1
    masks = masks.view(x.shape)
    ret = (masks,)
    if return_values:
        values = x.data[masks.bool()]
        ret = (values,) + ret
    if return_indices:
        ret = ret + (indexes,)
    return ret
    
def compress_topr(x, r):
    values, masks = topr(x, r)
    masks = pack_uint8_tensor(masks)
    return values, masks
    
def compress_topk(x, k, return_indices=False):
    if return_indices:
        values, masks, indices = topk(x, k, return_indices=return_indices)
        masks = pack_uint8_tensor(masks)
        return values, masks, indices
    else:
        values, masks = topk(x, k, return_indices=return_indices)
        masks = pack_uint8_tensor(masks)
        return values, masks

def decompress_topk(values, masks, original_shape):
    masks = unpack_uint8_tensor(masks)
    x = torch.zeros(masks.shape, dtype=values.dtype, device=values.device)
    x[masks.bool()] = values
    return x