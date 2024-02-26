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
import open3d as o3d
from joblib import Parallel, delayed
from tqdm import tqdm

from collections import defaultdict

# data_prefix = "/mnt/kostas-graid/datasets/ev_data"
data_prefix = "/new-pool"

SEQ_SIZE = 30
SEQ_LEN = 100000


# @profile
def get_ev(f_data, ms_map, CHUNK_SIZE, i):
    x_ev = f_data["prophesee"]["left"]["x"][
        ms_map[i * CHUNK_SIZE] : ms_map[(i + 1) * CHUNK_SIZE]
    ]
    y_ev = f_data["prophesee"]["left"]["y"][
        ms_map[i * CHUNK_SIZE] : ms_map[(i + 1) * CHUNK_SIZE]
    ]
    p_ev = f_data["prophesee"]["left"]["p"][
        ms_map[i * CHUNK_SIZE] : ms_map[(i + 1) * CHUNK_SIZE]
    ]
    t_ev = f_data["prophesee"]["left"]["t"][
        ms_map[i * CHUNK_SIZE] : ms_map[(i + 1) * CHUNK_SIZE]
    ]

    block_id = x_ev // 32 + (y_ev // 32) * 40

    cnts, bins = np.histogram(block_id, np.arange(920))

    ind = np.argpartition(cnts, -SEQ_SIZE)[-SEQ_SIZE:]
    if cnts[ind[-1]] < 100000:
        return None, False
    ev_out = np.zeros((SEQ_SIZE, SEQ_LEN, 4))
    for z in range(SEQ_SIZE):
        ev_seq = np.where(block_id == ind[z])[0]
        evs = np.stack([x_ev[ev_seq], y_ev[ev_seq], p_ev[ev_seq], t_ev[ev_seq]]).T

        if evs.shape[0] > SEQ_LEN:
            rnd_idx = np.random.choice(evs.shape[0], SEQ_LEN, replace=False)
            evs = evs[np.sort(rnd_idx)]
        else:
            evs = np.pad(evs, ((0, SEQ_LEN - evs.shape[0]), (0, 0)), mode="constant")

        ev_out[z] = evs

    return ev_out, True


def convert_file(seq_name):
    # File Path for M3ED Files
    data_path = f"{data_prefix}/events/m3ed/{seq_name}/{seq_name}_data.h5"

    # Load data in from pose files
    f_data = h5py.File(
        data_path,
        "r",
    )

    f_exp = h5py.File(f"{data_prefix}/ev_list/m3ed/{seq_name}.h5", "w")

    # Create Max Shape Logic and Resize at the End!
    CHUNK_SIZE = 500
    ms_map = f_data["prophesee"]["left"]["ms_map_idx"][:]  # not going touse l=
    total_filts = ((len(ms_map) // CHUNK_SIZE) + 1) * SEQ_SIZE
    ev_set_h5 = f_exp.create_dataset(
        "events",
        dtype=np.float32,
        shape=(total_filts, SEQ_LEN, 4),
        chunks=(1, SEQ_LEN, 4),
        **hdf5plugin.Blosc2(cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE),
    )

    idx = 0

    st = time.time()

    for i in tqdm(range((len(ms_map) // CHUNK_SIZE) - 1)):
        # for i in tqdm(range(20, 28)):
        ev_data, res = get_ev(f_data, ms_map, CHUNK_SIZE, i)
        if res:
            ev_set_h5[idx : idx + SEQ_SIZE] = ev_data
            idx += SEQ_SIZE

    print(time.time() - st, idx)
    f_exp.close()


def run(args):
    m3ed_directory = f"{data_prefix}/events/m3ed/"
    files_ls = os.listdir(m3ed_directory)
    files_ls.sort()
    files_ls = [file for file in files_ls if "car" in file]

    # for i in range(len(files_ls)):
    # try:

    convert_file(files_ls[args.num])
    # except:
    #     print(f"Error on {i}")
    #     import traceback

    #     traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", help="file_num", type=int, default=0)
    args = parser.parse_args()
    run(args)
