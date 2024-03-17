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


SEQ_SIZE = 12000000  # Sequences to be used in model
TIME_LEN = 1200  # How many ms to load in of data
res = (480, 640)


def get_ev(f_data, ms_map, TIME_LEN, i):
    x_ev = f_data["dvs"]["left"]["x"][ms_map[i * TIME_LEN] : ms_map[(i + 1) * TIME_LEN]]
    y_ev = f_data["dvs"]["left"]["y"][ms_map[i * TIME_LEN] : ms_map[(i + 1) * TIME_LEN]]
    p_ev = f_data["dvs"]["left"]["p"][ms_map[i * TIME_LEN] : ms_map[(i + 1) * TIME_LEN]]
    t_ev = f_data["dvs"]["left"]["t"][ms_map[i * TIME_LEN] : ms_map[(i + 1) * TIME_LEN]]

    ev_out = np.zeros((SEQ_SIZE, 4))
    evs = np.stack([np.clip(x_ev, 0, 639), np.clip(y_ev, 0, 479), p_ev, t_ev]).T
    ev_out[:] = evs[:SEQ_SIZE]

    return ev_out, True


def convert_file():
    # File Path for Spinner
    data_path = f"/new-pool/events/spinner/speed80.h5"

    # Load data in from pose files
    f_data = h5py.File(
        data_path,
        "r",
    )
    f_exp = h5py.File(
        "/new-pool/ev_list/spinner/train_spin_full.h5",
        "w",
    )
    times = f_data["dvs"]["left"]["t"][:]
    ms_map = np.searchsorted(times, np.arange(times[10], times[-1], 1000))

    # Create Max Shape Logic and Resize at the End!

    total_filts = (len(ms_map) // TIME_LEN) + 1
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
            ev_set_h5[idx : idx + 1, : len(ev_data)] = ev_data
            ev_set_h5[idx : idx + 1, len(ev_data) :] = 0
            idx += 1

    print(time.time() - st, idx)
    f_exp.close()


if __name__ == "__main__":
    convert_file()
