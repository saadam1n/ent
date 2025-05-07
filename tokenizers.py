import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch import Tensor

class GeneralTokenizer:
    """
    Implements any form of tokenization.
    """

    def __init__(self):
        pass

    def fit(self, bc_dataset : torch.utils.data.Dataset):
        raise NotImplementedError()
    
    def encode(self, data : list[Tensor]):
        raise NotImplementedError()
    
    def decode(self, data):
        raise NotImplementedError()

class DefaultTokenizer:
    """
    Implements the normal bin normalization. Values [-1, 1] are remapped to [0, 99] and a termination column is added
    """
    def __init__(self):
        pass

    def fit(self, bc_dataset : torch.utils.data.Dataset):
        print("fit() called on DefaultTokenizer. "
              "Please note that the default tokenizer has no trainable/fittable parameters and thus this function has not effect.")

    def encode(self, data : list[Tensor]):
        pass

    def decode(self, data):
        pass