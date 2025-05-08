import torch
import trajtok

import h5py
import numpy as np

import random
import os
import bisect

class BehaviorColoningDataset(torch.utils.data.Dataset):
    """
    This dataset loads the raw robomimic trajectories into memory.
    It allows for training, somehow. I have yet to figure that out.
    For now I will get the transformer to predict a single action chunk at a time.
    """
    def __init__(self, tokenizer : trajtok.GeneralTokenizer, history_size : int):
        super(BehaviorColoningDataset, self).__init__()

        self.tokenizer = tokenizer
        self.chunk_size = self.tokenizer.chunk_size if hasattr(self.tokenizer, "chunk_size") else 20
        self.history_size = history_size

        self.load_dataset()

        print("Fitting tokenizer... this may take a while!")
        self.tokenizer.fit(self)

        print("Fitting completed. BehaviorColoningDataset is ready to use for training policies.")

    def load_dataset(self):
        robomimic_files = ["low_dim_v141(4).hdf5", "low_dim_v141(3).hdf5", "low_dim_v141(2).hdf5", "low_dim_v141(1).hdf5", "low_dim_v141.hdf5"]

        self.train_samples = []
        self.test_samples = []

        total_num_demos = 0

        print("Iterating through HDF5 files...")
        for hdf5_file in robomimic_files:
            actual_path = os.path.join("/data/oldsotavla/", hdf5_file)

            current_num_demos = 0
            with h5py.File(actual_path, 'r') as f:
                for demo in f['data']:
                    current_num_demos += 1

                    is_train_sample = random.random() < 0.95

                    demo_pos = f['data'][demo]['obs']['robot0_eef_pos'][:]
                    demo_rot = f['data'][demo]['obs']['robot0_eef_quat'][:]

                    demo_eef_actions = torch.from_numpy(np.concatenate((demo_pos, demo_rot), axis=1)).float()
                    unpadded_size = demo_eef_actions.shape[0]

                    # for train: a window can start from any point in the trajectory
                    # for eval: break traj into different points
                    padding_size = self.chunk_size - 1 if is_train_sample else ((unpadded_size - 1) // self.chunk_size + 1) * self.chunk_size - unpadded_size

                    if padding_size != 0:
                        padded_actions = torch.cat(
                            (
                                demo_eef_actions,
                                torch.zeros(padding_size, *demo_eef_actions.shape[1:])
                            ),
                            dim=0
                        )
                    else:
                        padded_actions = demo_eef_actions

                    if not is_train_sample and padded_actions.shape[0] % self.chunk_size != 0:
                        raise RuntimeError(f"Bad padding! Original shape: {demo_eef_actions.shape}, padded shape: {padded_actions.shape}, chunk size {self.chunk_size}")
                    
                    prepadded_shape = torch.cat(
                        (
                            padded_actions, 
                            torch.zeros(self.history_size, *padded_actions.shape[1:])
                        ),
                        dim=0
                    )

                    output = self.train_samples if is_train_sample else self.test_samples

                    output.append((prepadded_shape, unpadded_size))

                    

            print(f"\t{hdf5_file} had {current_num_demos} num demos")
            total_num_demos += current_num_demos


        self.train_mode = True
        self.tokenization = True

        print(f"Total demo count is {total_num_demos}")
        print(f"\tTrain size is: {len(self.train_samples)}")
        print(f"\tTest size is {len(self.test_samples)}")
        print(f"\tTest/train split is {100.0 * len(self.train_samples) / total_num_demos}")

        self.normalize_chunks()

        # now, we count the number of windows and map everything
        # ignore test for now because that will cause me a headache

        self.train_window_offsets = [
            len(a) - self.history_size - self.chunk_size + 1 for a, _ in self.train_samples
        ]

        for i in range(1, len(self.train_window_offsets)):
            self.train_window_offsets[i] += self.train_window_offsets[i - 1]

        self.train_window_count = self.train_window_offsets[-1]

        print(f"Number of trainable sliding windows in dataset is {self.train_window_count}")
        

    """
    As opposed to normal mean/variance normalization, we do FAST-style normalization which relies on compressing the 1st and 99th ranges to [-1, 1]
    """
    def normalize_chunks(self):
        # don't include test samples to help "isolate" the normalizaiton
        all_actions = [
            a for a, _ in self.train_samples
        ]

        all_actions = torch.cat(all_actions, dim=0)

        sorted_chunks, _ = torch.sort(all_actions, dim=0)

        lower = sorted_chunks[int(0.01 * sorted_chunks.shape[0])]
        upper = sorted_chunks[int(0.99 * sorted_chunks.shape[0])]
        width = upper - lower

        print("Dataset statistics:")
        print(f"\tlower: {lower}")
        print(f"\tupper: {upper}")
        print(f"\twidth: {width}")

        inv_width = 2.0 / width

        for i in range(len(self.train_samples)):
            self.train_samples[i][0].sub_(lower).mul_(inv_width).sub_(1.0)

        for i in range(len(self.test_samples)):
            self.test_samples[i][0].sub_(lower).mul_(inv_width).sub_(1.0)
    
    def fetch_train_sample(self, idx):
        demo_idx = bisect.bisect_right(self.train_window_offsets, idx)

        prev_sum = self.train_window_offsets[demo_idx - 1] if demo_idx > 0 else 0
        offset = idx - prev_sum

        traj = self.train_samples[demo_idx][0][offset:offset + self.chunk_size + self.history_size]

        pre = traj[:self.history_size]
        tok = traj[self.history_size:]

        tok = self.tokenizer.encode(tok) if self.tokenization else tok

        #print(f"Returning items of shape {pre.shape}\t{tok.shape}")

        pre = pre.to("cuda")
        tok = tok.to("cuda")

        return pre, tok
    
    def __len__(self):
        return self.train_window_count
    
    def __getitem__(self, index):
        if self.train_mode:
            return self.fetch_train_sample(index)
        else:
            raise NotImplementedError()