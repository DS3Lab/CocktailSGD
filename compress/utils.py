import torch
import cupy
import numpy as np
from torch.utils.dlpack import to_dlpack, from_dlpack

def cupy_to_tensor(x):
    return from_dlpack(x.toDlpack())

def tensor_to_cupy(x):
    return cupy.fromDlpack(to_dlpack(x))

def pack_uint8_tensor(x):
    if x.device != torch.device('cpu'):
        return cupy_to_tensor(
            cupy.packbits(tensor_to_cupy(x))
        )
    else:
        return torch.from_numpy(
            np.packbits(x.numpy())
        )

def unpack_uint8_tensor(x):
    if x.device != torch.device('cpu'):
        return cupy_to_tensor(
            cupy.unpackbits(tensor_to_cupy(x))
        )
    else:
        return torch.from_numpy(
            np.unpackbits(x.numpy())
        )

def pack_low_bit_tensor(x, bits):
    
    if x.device != torch.device('cpu'):
        assert x.dtype == torch.uint8
        y = cupy.packbits(
            cupy.unpackbits(tensor_to_cupy(x)).reshape(*x.shape, 8)[..., -bits:]
        )
        y = cupy_to_tensor(y)
    else:
        y = np.packbits(
            np.unpackbits(x.numpy()).reshape(*x.shape, 8)[..., -bits:]
        )
        y = torch.from_numpy(y)
        
    return y

def unpack_low_bit_tensor(x, bits, original_shape):
    
    if x.device != torch.device('cpu'):
        y = cupy.packbits(cupy.pad(
            cupy.unpackbits(
                tensor_to_cupy(x)
            )[:np.prod(original_shape)*bits].reshape(-1, bits),
            ((0,0), (8-bits, 0))
        ))
        y = cupy_to_tensor(y).view(original_shape)
    else:
        y = np.packbits(np.pad(
            np.unpackbits(
                x.numpy()
            )[:np.prod(original_shape)*bits].reshape(-1, bits),
            ((0,0), (8-bits, 0))
        ))
        y = torch.from_numpy(y).view(original_shape)
    return y


def pin_memory(array):
    mem = cupy.cuda.alloc_pinned_memory(array.nbytes)
    ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    ret[...] = array
    return ret