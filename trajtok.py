import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch import Tensor
from transformers import AutoProcessor

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
        
        if True:
            print("Skipping DefaultTokenizer tests. If you want the tests to run, please change this line in the code.")
            return

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

        self.action_dim_cached = data.shape[-1]

        tform_data = data * 0.5 + 0.5
        round_data = (tform_data * self.token_count).long().clamp_(min=0, max=self.token_count - 1)

        round_data = round_data.flatten(-2)

        return round_data

    def decode(self, data : Tensor):
        float_data = data.float() / self.token_count
        tform_data = float_data * 2.0 - 1.0

        tform_data.unflatten(-1, (-1, self.action_dim_cached))

        return tform_data
    

class FastTokenizer:
    """
    Implements the normal bin normalization. Values [-1, 1] are remapped to [0, 99] and a termination column is added
    """
    def __init__(self):
        super().__init__()

        self.token_count = 1025
        self.tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)

    def fit(self, bc_dataset : torch.utils.data.Dataset):
        print("fit() called on FastTokenizer. Please be aware that tokenizing all windows will likely take a massive chunk of time.")

        bc_dataset.tokenization = False

        bc_dataloader = torch.utils.data.DataLoader(
            bc_dataset,
            batch_size=4096,
            shuffle=True
        )

        concat_chunks = []
        for i, data in enumerate(bc_dataloader):
            print(f"\tLoading batch {i}")

            _, traj = data

            concat_chunks.append(traj)

        concat_chunks = torch.cat(concat_chunks, dim=0)

        self.tokenizer = self.tokenizer.fit(concat_chunks.detach().cpu())

        print(f"L1 loss over all windows was {F.l1_loss(self.decode(self.encode(concat_chunks)), concat_chunks)}")

        bc_dataset.tokenization = True

    def encode(self, data : Tensor):
        tokenized = self.tokenizer(data.detach().cpu())

        if len(tokenized) > 1024:
            print(f"Mass tokenization detected! "
                  f"Average length of everything tokenized was {sum([len(a) for a in tokenized]) / len(tokenized)}. "
                  f"Maximum length was {max([len(a) for a in tokenized])}. ")

        self.cached_max_length = data.shape[-2] * data.shape[-1]

        # pad all tokenized elements
        padded_tokenized = [
            self.pad_sequence(a) for a in tokenized
        ]

        pt_tensor = torch.tensor(padded_tokenized, dtype=torch.long).squeeze()

        return pt_tensor

    def pad_sequence(self, sequence : list[int]):      
        if len(sequence) < self.cached_max_length:
            sequence.append(self.token_count - 1)

        sequence += [-1] * (self.cached_max_length - len(sequence))

        return sequence

    def depad_sequence(self, pt_list : np.ndarray):
        end_idx = -1
        while pt_list[end_idx] == -1 or pt_list[end_idx] == self.token_count - 1:
            end_idx -= 1

        end_idx += 1

        return pt_list[:end_idx] if end_idx != 0 else pt_list
    
    def decode(self, data : Tensor):
        data_np = data.detach().cpu().numpy()

        all_sequences = [
            self.depad_sequence(data_np[i]) for i in range(data.shape[0])
        ]

        detokenized = self.tokenizer.decode(all_sequences)

        return torch.from_numpy(detokenized).to("cuda")