import time
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
from lightning.pytorch.callbacks import BasePredictionWriter, ModelCheckpoint


class DataWriter(BasePredictionWriter):
    def __init__(
        self,
        write_interval="batch",
        total_data=(100, 100),
        res=(360, 360),
        cfg=None,
        i=1,
    ):
        super().__init__(write_interval)
        Path(f"main/inf/{cfg.exp}/").mkdir(parents=True, exist_ok=True)

        self.f_w = h5py.File(
            f"main/inf/{cfg.exp}/{i:05}.h5",
            "w",
        )

        tot_pred = (int((cfg.end_time - cfg.time_history) // cfg.time_dt) - 2) - 5

        self.train = self.f_w.create_group("train")
        self.train_y = self.train.create_dataset(
            "y",
            (total_data[1], tot_pred, res[0], res[1]),
            np.float16,
            chunks=(1, tot_pred, res[0], res[1]),
            **hdf5plugin.Blosc2(
                cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE
            ),
        )
        self.train_yhat = self.train.create_dataset(
            "y_hat",
            (total_data[1], tot_pred, res[0], res[1]),
            np.float16,
            chunks=(1, tot_pred, res[0], res[1]),
            **hdf5plugin.Blosc2(
                cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE
            ),
        )
        self.train_x = self.train.create_dataset(
            "x",
            (total_data[1], tot_pred, cfg.in_chan, res[0], res[1]),
            np.float16,
            chunks=(1, tot_pred, cfg.in_chan, res[0], res[1]),
            **hdf5plugin.Blosc2(
                cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE
            ),
        )
        self.val = self.f_w.create_group("val")
        self.val_y = self.val.create_dataset(
            "y",
            (total_data[0], tot_pred, res[0], res[1]),
            np.float16,
            chunks=(1, tot_pred, res[0], res[1]),
            **hdf5plugin.Blosc2(
                cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE
            ),
        )
        self.val_yhat = self.val.create_dataset(
            "y_hat",
            (total_data[0], tot_pred, res[0], res[1]),
            np.float16,
            chunks=(1, tot_pred, res[0], res[1]),
            **hdf5plugin.Blosc2(
                cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE
            ),
        )
        self.val_x = self.val.create_dataset(
            "x",
            (total_data[0], tot_pred, cfg.in_chan, res[0], res[1]),
            np.float16,
            chunks=(1, tot_pred, cfg.in_chan, res[0], res[1]),
            **hdf5plugin.Blosc2(
                cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE
            ),
        )

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        if dataloader_idx == 0:
            self.val_yhat[batch_indices[0] : batch_indices[-1] + 1] = prediction[0]
            self.val_y[batch_indices[0] : batch_indices[-1] + 1] = prediction[1]
            self.val_x[batch_indices[0] : batch_indices[-1] + 1] = prediction[2]

        else:
            self.train_yhat[batch_indices[0] : batch_indices[-1] + 1] = prediction[0]
            self.train_y[batch_indices[0] : batch_indices[-1] + 1] = prediction[1]
            self.train_x[batch_indices[0] : batch_indices[-1] + 1] = prediction[2]
