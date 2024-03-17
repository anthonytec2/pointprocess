from DeepSTPP.src.model import *
import torch

import lightning as pl
from omegaconf import OmegaConf
from torch import nn
import hydra
import sys
from mamba_ssm.models.mixer_seq_simple import *
from DeepSTPP.src.model import Decoder
import matplotlib.pyplot as plt
from torch.nn import functional as F
from datawriter import DataWriter

pl.seed_everything(1)

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True

idm = 0


class STCNNModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.stconv = torch.nn.Conv2d(
            in_channels=cfg.in_chan,
            out_channels=cfg.hidden,
            kernel_size=(cfg.kernel, cfg.kernel),
            padding="same",
        )
        self.combine = torch.nn.Conv2d(
            in_channels=cfg.hidden, out_channels=1, kernel_size=(1, 1), padding="same"
        )

        self.save_hyperparameters()
        self.loss = nn.BCELoss()

    def forward(self, x):
        x = self.stconv(x)
        x = self.combine(x)
        x = F.sigmoid(x)
        return x

    def model_step(self, batch, name="train"):
        (curr_vol, pred_vol) = batch
        est_vol = self(curr_vol)

        loss = self.loss(est_vol, pred_vol[:, 0][:, None, :, :])
        for i in range(1, cfg.pred_horizon):
            curr_vol = torch.hstack([curr_vol[:, 1:], est_vol])
            est_vol = self(curr_vol)
            loss += self.loss(est_vol, pred_vol[:, i][:, None, :, :])

        if self.trainer.global_step % self.cfg.interval == 0:
            with torch.no_grad():
                fig = plt.figure(figsize=(12, 12))
                for i in range(6):
                    plt.subplot(6, 4, 4 * i + 1)

                    plt.imshow(
                        (torch.sum(curr_vol[i, -10:], axis=0)).cpu().detach().numpy(),
                        vmin=0,
                        # vmax=1,
                    )
                    # torch.sum(
                    #         curr_vol[i] * torch.arange(90)[:, None, None].cuda(), axis=0
                    #     )
                    plt.title("Prior T")

                    plt.subplot(6, 4, 4 * i + 2)
                    plt.imshow(
                        torch.sum(est_vol[i], axis=0).cpu().detach().numpy(),
                        vmin=0,
                        vmax=1,
                    )
                    plt.title("NN Estimate T+1")

                    plt.subplot(6, 4, 4 * i + 3)
                    plt.imshow(
                        (pred_vol[i, -1]).cpu().detach().numpy(),
                        vmin=0,
                        vmax=1,
                    )
                    plt.title("GT T+1")

                    plt.subplot(6, 4, 4 * i + 4)

                    plt.imshow(
                        ((pred_vol[i, -1] - est_vol[i, 0]) ** 2).cpu().detach().numpy(),
                        vmin=0,
                        # vmax=1,
                    )
                    plt.title("Res T+1")

            self.logger.experiment.add_figure(
                "Events",
                fig,
                global_step=self.trainer.global_step,
            )
            plt.close("all")

        with torch.no_grad():
            if not name == "pred":
                self.log_dict(
                    {
                        f"{name}_tot": loss,
                    },
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

        return loss

    def pred_stp(self, batch, name="train"):
        (curr_vol, pred_vol) = batch
        est_vol = self(curr_vol)
        est_vols = [est_vol]
        curr_vols = [curr_vol]

        for i in range(1, 20):
            curr_vol = torch.hstack([curr_vol[:, 1:], est_vol])
            curr_vols.append(curr_vol)
            est_vol = self(curr_vol)
            est_vols.append(est_vol)

        return est_vols, pred_vol, curr_vols

    def validation_step(self, batch, batch_idx):
        loss = self.model_step(batch, name="val")
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.model_step(batch, name="train")

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        y_hat, y, x = self.pred_stp(batch, "pred")

        return torch.hstack(y_hat), y, torch.stack(x, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.cfg.lr, fused=True)
        return optimizer
