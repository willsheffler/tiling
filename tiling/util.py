import sys
import numpy as np

def copy_array(a):
    if 'torch' in sys.modules:
        import torch
        if torch.is_tensor(a): return a.clone()
    return a.copy()

def cast_uint8(a):
    if 'torch' in sys.modules:
        import torch
        if torch.is_tensor(a): return a.to(torch.uint8)
    return a.astype(np.uint8)
