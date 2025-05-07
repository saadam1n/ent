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
        self.token_count = -1

    def fit(self, bc_dataset : torch.utils.data.Dataset):
        raise NotImplementedError()
    
    def encode(self, data : Tensor):
        raise NotImplementedError()
    
    def decode(self, data):
        raise NotImplementedError()

class DefaultTokenizer:
    """
    Implements the normal bin normalization. Values [-1, 1] are remapped to [0, 99] and a termination column is added
    """
    def __init__(self):
        super().__init__()

        self.token_count = 100

    def fit(self, bc_dataset : torch.utils.data.Dataset):
        print("fit() called on DefaultTokenizer. "
              "Please note that the default tokenizer has no trainable/fittable parameters and thus this function has no effect. "
              "Instead, this function will compute the loss on various trajectories to ensure the tokenizer is working as intended. ")
        
        bc_dataset.tokenization = False

        bc_dataloader = torch.utils.data.DataLoader(
            bc_dataset,
            batch_size=4096,
            shuffle=True
        )

        num_batches = 0
        running_loss = 0
        for i, data in enumerate(bc_dataloader):
            print(f"\tProcessing batch {i}")

            _, traj = data

            rct = self.decode(self.encode(traj))

            loss = F.l1_loss(rct, traj)

            running_loss = running_loss + loss
            num_batches += 1

        running_loss = running_loss / num_batches

        print(f"Total L1 loss was {running_loss}")

        bc_dataset.tokenization = True

    def encode(self, data : Tensor):
        tform_data = data * 0.5 + 0.5
        round_data = (tform_data * self.token_count).int().clamp_(min=0, max=self.token_count - 1)

        return round_data

    def decode(self, data : Tensor):
        float_data = data.float() / self.token_count
        tform_data = float_data * 2.0 - 1.0

        return tform_data