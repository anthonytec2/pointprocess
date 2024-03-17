from pathlib import Path
from dl import PPDl
import h5py
import hdf5plugin
import lightning as pl
import numpy as np
import yaml
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm


class PPDataModule(pl.LightningDataModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.batch_size = cfg.batch_size
        self.cfg = cfg
        self.res = (32, 32)

    def setup(self, stage: str):
        if stage == "test":
            self.shuf = False
        else:
            self.shuf = False
        self.train_loader = PPDl(self.cfg, True)
        self.val_loader = PPDl(self.cfg, False)

    def train_dataloader(self):
        return DataLoader(
            self.train_loader,
            batch_size=self.batch_size,
            shuffle=self.shuf,
            num_workers=self.cfg.workers,
            # persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_loader,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.workers,
            # persistent_workers=True,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.train_loader,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.workers,
            # persistent_workers=True,
            pin_memory=True,
        )
