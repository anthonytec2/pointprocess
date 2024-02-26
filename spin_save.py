import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import h5py
import hdf5plugin
import numba
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from collections import defaultdict


SEQ_SIZE = 19000  # Sequences to be used in model
TIME_LEN = 300  # How many ms to load in of data
PS = 32
res = (480, 640)
TOT_SAVE = 100  # Number of patches to use
# Load in an a block of event time (t, t+delta)
# Cut up the world into patches and create sequence
# Record all the sequences to an h5 file


# @profile
def get_ev(f_data, ms_map, TIME_LEN, i):
    x_ev = f_data["dvs"]["left"]["x"][ms_map[i * TIME_LEN] : ms_map[(i + 1) * TIME_LEN]]
    y_ev = f_data["dvs"]["left"]["y"][ms_map[i * TIME_LEN] : ms_map[(i + 1) * TIME_LEN]]
    p_ev = f_data["dvs"]["left"]["p"][ms_map[i * TIME_LEN] : ms_map[(i + 1) * TIME_LEN]]
    t_ev = f_data["dvs"]["left"]["t"][ms_map[i * TIME_LEN] : ms_map[(i + 1) * TIME_LEN]]

    block_id = x_ev // PS + (y_ev // PS) * (res[1] // PS)

    cnts, bins = np.histogram(block_id, np.arange(300))

    ind = np.argpartition(cnts, -TOT_SAVE)[-TOT_SAVE:]

    ev_out = np.zeros((TOT_SAVE, SEQ_SIZE, 4))
    cnt = 0
    for z in range(TOT_SAVE):
        ev_seq = np.where(block_id == ind[z])[0]
        evs = np.stack([x_ev[ev_seq], y_ev[ev_seq], p_ev[ev_seq], t_ev[ev_seq]]).T

        if evs.shape[0] > SEQ_SIZE:
            rnd_idx = np.random.choice(evs.shape[0], SEQ_SIZE, replace=False)
            evs = evs[np.sort(rnd_idx)]
        else:
            evs = np.pad(evs, ((0, SEQ_SIZE - evs.shape[0]), (0, 0)), mode="constant")

        ev_out[cnt] = evs
        cnt += 1

    return ev_out[:cnt], True


def convert_file():
    # File Path for Spinner
    data_path = f"/new-pool/events/spinner/speed80.h5"

    # Load data in from pose files
    f_data = h5py.File(
        data_path,
        "r",
    )
    f_exp = h5py.File(
        "/new-pool/ev_list/spinner/spin.h5",
        "w",
    )
    times = f_data["dvs"]["left"]["t"][:]
    ms_map = np.searchsorted(times, np.arange(times[10], times[-1], 1000))

    # Create Max Shape Logic and Resize at the End!

    total_filts = ((len(ms_map) // TIME_LEN) + 1) * TOT_SAVE
    ev_set_h5 = f_exp.create_dataset(
        "events",
        dtype=np.float32,
        shape=(total_filts, SEQ_SIZE, 4),
        chunks=(1, SEQ_SIZE, 4),
        **hdf5plugin.Blosc2(cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE),
    )

    idx = 0

    st = time.time()

    for i in tqdm(range((len(ms_map) // TIME_LEN) - 1)):
        ev_data, res = get_ev(f_data, ms_map, TIME_LEN, i)
        if res:
            ev_set_h5[idx : idx + len(ev_data)] = ev_data
            idx += len(ev_data)

    print(time.time() - st, idx)
    f_exp.close()


if __name__ == "__main__":
    convert_file()
