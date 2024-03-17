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
            h5_file = h5py.File(cfg.train_file, "r")
        else:
            h5_file = h5py.File(cfg.test_file, "r")

        self.events = h5_file["events"][:]

        self.ev_len = np.where(np.sum(self.events, axis=(1, 2)) < 1e-2)[0][
            0
        ]  # len(self.events)

    def __getitem__(self, index):
        evs = self.events[index]

        # if evs[-1, 3] < 1e-2:
        #     cut_pt = np.where(evs[:, 0] == 0)[0][0]

        #     evs[:cut_pt, 3] = (evs[:cut_pt, 3] - evs[0, 3]) / 5e5
        #     evs[:, 2] = 0
        #     evs[:cut_pt, 2] = 1
        # else:

        if self.cfg.model == "pp":
            evs[:, 3] = (evs[:, 3] - evs[0, 3]) / 5e5
            evs[:, 2] = 1

            evs[:, 0] = (evs[:, 0] % 32) / 32
            evs[:, 1] = (evs[:, 1] % 32) / 32

            return (
                evs[:, 0].astype(np.float16),
                evs[:, 1].astype(np.float16),
                evs[:, 3],
                evs[:, 2].astype(bool),
            )
        elif self.cfg.model == "stcnn":
            time = evs[:, 3] - evs[0, 3]

            start_time = 0.24
            t_idx1 = np.searchsorted(time, start_time * 1e6)  # 10

            curr_frame = np.zeros((90, 32, 32), dtype=np.float32)

            curr_frame[
                (time[:t_idx1] // 3000).astype(np.uint16),
                (evs[:t_idx1, 1] % 32).astype(np.uint16),
                (evs[:t_idx1, 0] % 32).astype(np.uint16),
            ] = 1
            t_prev = t_idx1
            pred_frame = np.zeros((20, 32, 32), dtype=np.float32)
            for i in range(1, 21):
                t_new = np.searchsorted(time, (start_time + 0.003 * i) * 1e6)
                pred_frame[
                    i - 1,
                    (evs[t_prev:t_new, 1] % 32).astype(np.uint16),
                    (evs[t_prev:t_new, 0] % 32).astype(np.uint16),
                ] = 1
                t_prev = t_new

            return curr_frame, pred_frame
        elif self.cfg.model == "ftcnn":
            time = evs[:, 3] - evs[0, 3]

            start_time = self.cfg.time_history
            t_idx1 = np.searchsorted(time, start_time * 1e6)  # 10

            tot_hist = int((start_time) // self.cfg.time_dt) + 1
            curr_frame = np.zeros(
                (tot_hist, int(self.cfg.res[0]), int(self.cfg.res[1])), dtype=np.float32
            )

            curr_frame[
                (time[:t_idx1] // (self.cfg.time_dt * 1e6)).astype(np.uint16),
                (evs[:t_idx1, 1]).astype(np.uint16),
                (evs[:t_idx1, 0]).astype(np.uint16),
            ] = 1
            t_prev = t_idx1

            tot_pred = (
                int((self.cfg.end_time - self.cfg.time_history) // self.cfg.time_dt) - 2
            )
            pred_frame = np.zeros(
                (tot_pred, int(self.cfg.res[0]), int(self.cfg.res[1])), dtype=np.float32
            )
            for i in range(1, tot_pred):
                t_new = np.searchsorted(time, (start_time + self.cfg.time_dt * i) * 1e6)
                pred_frame[
                    i - 1,
                    (evs[t_prev:t_new, 1]).astype(np.uint16),
                    (evs[t_prev:t_new, 0]).astype(np.uint16),
                ] = 1

                t_prev = t_new
            # Fix This
            pred_frame = pred_frame[:-5]

            return curr_frame, pred_frame

    def __len__(self):
        return self.ev_len
