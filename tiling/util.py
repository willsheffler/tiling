import sys
import numpy as np

def copy_array(a):
    if 'torch' in sys.modules:
        import torch
        if torch.is_tensor(a): return a.clone()
    return a.copy()
