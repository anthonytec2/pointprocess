import os
import sys

import h5py
import hdf5plugin
import numpy as np
import torch


class PPDl(torch.utils.data.Dataset):
    def __init__(self, cfg, train):
        super(PPDl, self).__init__()
        self.cfg = cfg
        if train:
            h5_file = h5py.File(cfg.train_file)
        else:
            h5_file = h5py.File(cfg.test_file)

        self.events = h5_file["events"][:]

        self.ev_len = np.where(np.sum(self.events, axis=(1, 2, 3)) < 1e-2)[0][
            0
        ]  # len(self.events)

    def __getitem__(self, index):
        evs = self.events[index, 0]

        if evs[-1, 3] < 1e-2:
            cut_pt = np.where(evs[:, 0] == 0)[0][0]

            evs[:cut_pt, 3] = (evs[:cut_pt, 3] - evs[0, 3]) / 5e5
            evs[:, 2] = 0
            evs[:cut_pt, 2] = 1
        else:
            evs[:, 3] = (evs[:, 3] - evs[0, 3]) / 5e5
            evs[:, 2] = 1

        evs[:, 0] = (evs[:, 0] % 32) / 32
        evs[:, 1] = (evs[:, 1] % 32) / 32

        return (
            evs[:, 0].astype(np.uint8),
            evs[:, 1].astype(np.uint8),
            evs[:, 3],
            evs[:, 2].astype(bool),
        )

    def __len__(self):
        return self.ev_len
