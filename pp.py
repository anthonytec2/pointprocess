from DeepSTPP.src.model import *
import torch

import lightning as pl
from omegaconf import OmegaConf
from torch import nn
import hydra
import sys
from mamba_ssm.models.mixer_seq_simple import *

pl.seed_everything(1)

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True

# Originally From: https://github.com/Rose-STL-Lab/DeepSTPP/blob/master/src/model.py#L179
"""
Log likelihood of no events happening from t_n to t
- ∫_{t_n}^t λ(t') dt' 

tn_ti: [batch, seq_len]
t_ti: [batch, seq_len]
w_i: [batch, seq_len]
b_i: [batch, seq_len]

return: scalar
"""


def ll_no_events(w_i, b_i, tn_ti, t_ti):
    return torch.sum(w_i / b_i * (torch.exp(-b_i * t_ti) - torch.exp(-b_i * tn_ti)), -1)


def log_ft(t_ti, tn_ti, w_i, b_i):
    return ll_no_events(w_i, b_i, tn_ti, t_ti) + torch.log(t_intensity(w_i, b_i, t_ti))


"""
Compute spatial/temporal/spatiotemporal intensities

tn_ti: [batch, seq_len]
s_diff: [batch, seq_len, 2]
inv_var = [batch, seq_len, 2]
w_i: [batch, seq_len]
b_i: [batch, seq_len]

return: λ(t) [batch]
return: f(s|t) [batch] 
return: λ(s,t) [batch]
"""


def t_intensity(w_i, b_i, t_ti):
    v_i = w_i * torch.exp(-b_i * t_ti)
    lamb_t = torch.sum(v_i, -1)
    return lamb_t


def s_intensity(w_i, b_i, t_ti, s_diff, inv_var):
    v_i = w_i * torch.exp(-b_i * t_ti)
    v_i = v_i / torch.sum(v_i, -1).unsqueeze(-1)  # normalize
    g2 = torch.sum(s_diff * inv_var * s_diff, -1)
    g2 = torch.sqrt(torch.prod(inv_var, -1)) * torch.exp(-0.5 * g2) / (2 * np.pi)
    f_s_cond_t = torch.sum(g2 * v_i, -1)
    return f_s_cond_t


def intensity(w_i, b_i, t_ti, s_diff, inv_var):
    return t_intensity(w_i, b_i, t_ti) * s_intensity(w_i, b_i, t_ti, s_diff, inv_var)


class PPModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mamba = MixerModel(
            d_model=cfg.d_model,
            n_layer=cfg.n_layer,
            vocab_size=1,
            fused_add_norm=True,
            residual_in_fp32=True,
            rms_norm=True,
            device="cuda",
            dtype=torch.float16,
            ssm_cfg={
                "d_conv": cfg.d_conv,
                "d_state": cfg.d_state,
                "expand": cfg.expand,
            },
        )
        self.back_pts = nn.Parameter(torch.rand((cfg.num_pts, 2)))

        self.save_hyperparameters()

    def forward(self, x):
        out = self.mamba(x)
        # Predict W, sigmaS, sigmaT
        W = F.softplus(out[:, :, 0])
        sigmaS = F.softplus(out[:, :, 1])
        sigmaT = out[:, :, 2]

        return W, sigmaS, sigmaT

    def model_step(self, batch, name="train"):
        (x, y, t, mask) = batch

        seq_data = torch.stack(batch[:3], axis=2)

        W, sigmaS, sigmaT = self(seq_data)
        loss = 0
        batch_len = x.shape[0]
        for i in range(seq_data.shape[1]):
            st_x = seq_data
            st_y = seq_data[:, i, :].unsqueeze(1)
            background = self.back_pts.repeat(batch_len, 1, 1)
            t_cum = torch.cumsum(st_x[:, :, 2], -1)

            tn_ti = t_cum[:, -1:] - t_cum  # t_n - t_i
            tn_ti = torch.cat(
                (tn_ti, torch.zeros(batch_len, self.cfg.num_pts).cuda()), -1
            )
            import pdb

            pdb.set_trace()
            t_ti = tn_ti + st_y[:, :, 2]  # t - t_i

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

    def validation_step(self, batch, batch_idx):
        loss = self.model_step(batch, name="val")
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.model_step(batch, name="train")

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self.model_step(batch, "pred")

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.cfg.lr, fused=True)
        return optimizer
