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

pl.seed_everything(1)

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True

idm = 0
rnd_plot = torch.randint(low=0, high=4, size=(4,))
rnd_time = torch.randint(low=2000, high=10000, size=(4,))
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
            dtype=torch.float32,
            ssm_cfg={
                "d_conv": cfg.d_conv,
                "d_state": cfg.d_state,
                "expand": cfg.expand,
            },
        )
        output_dim = cfg.seq_len + cfg.num_pts
        self.back_pts = nn.Parameter(torch.rand((cfg.num_pts, 2)))
        self.w_dec = Decoder(cfg, out_dim=1, softplus=True)
        self.b_dec = Decoder(cfg, out_dim=1, softplus=False)
        self.s_dec = Decoder(cfg, out_dim=2, softplus=True)

        self.s_dec = Decoder(cfg, out_dim=2, softplus=True)

        self.save_hyperparameters()

    def forward(self, x):
        out = self.mamba(x)
        # Predict W, sigmaS, sigmaT

        W = self.w_dec(out)
        sigmaS = self.s_dec(out)
        sigmaT = self.b_dec(out)

        return W, torch.stack((1 / sigmaS[:, :, 0], 1 / sigmaS[:, :, 1]), -1), sigmaT

    def model_step(self, batch, name="train"):
        (x, y, t, mask) = batch

        seq_data = torch.stack(batch[:3], axis=2)
        t_pts = (
            torch.zeros((self.back_pts.shape[0], 1))
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1)
        )
        background = self.back_pts.unsqueeze(0).repeat(x.shape[0], 1, 1)
        back_int = torch.dstack([background.cuda(), t_pts.cuda()])
        seq_full = torch.hstack([seq_data, back_int])

        W, sigmaS, sigmaT = self(seq_full)

        loss = 0
        sll_tot = 0
        tll_tot = 0
        batch_len = x.shape[0]
        for z in range(1000):
            # for i in range(10000, seq_data.shape[1]):
            i = 10000 + np.random.randint(8000)
            st_x = seq_data[:, i - 10000 : i]
            st_y = seq_data[:, i, :].unsqueeze(1)

            s_diff = st_y[:, :, :2] - torch.cat((st_x[:, :, :2], background), 1)
            tn_ti = st_x[:, -1, -1][:, None] - st_x[:, :, -1]  # t_n - t_i
            tn_ti = torch.cat(
                (tn_ti, torch.zeros(batch_len, self.cfg.num_pts).cuda()), -1
            )
            t_ti = st_y[:, :, -1] - torch.cat(
                (st_x[:, :, -1], torch.zeros(batch_len, self.cfg.num_pts).cuda()), -1
            )
            w_sub = torch.hstack([W[:, i - 10000 : i, 0], W[:, seq_data.shape[1] :, 0]])
            sigmaS_sub = torch.hstack(
                [sigmaS[:, i - 10000 : i], sigmaS[:, seq_data.shape[1] :]]
            )
            sigmaT_sub = torch.hstack(
                [sigmaT[:, i - 10000 : i, 0], sigmaT[:, seq_data.shape[1] :, 0]]
            )

            sll = torch.log(s_intensity(w_sub, sigmaT_sub, t_ti, s_diff, sigmaS_sub))
            tll = log_ft(t_ti, tn_ti, w_sub, sigmaT_sub)
            loss += -sll.mean() - tll.mean()
            sll_tot += -sll.mean()
            tll_tot += -tll.mean()

        with torch.no_grad():
            if not name == "pred":
                self.log_dict(
                    {
                        f"{name}_tot": loss,
                        f"{name}_sll": sll_tot,
                        f"{name}_tll": tll_tot,
                    },
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
        if self.trainer.global_step % self.cfg.interval == 0:
            with torch.no_grad():
                rnd_seq = seq_data[rnd_plot]

                fig = plt.figure(figsize=(12, 12))
                for i in range(4):
                    plt.subplot(4, 3, 3 * i + 1)
                    seq = rnd_seq[i, rnd_time[i] - 1000 : rnd_time[i]]
                    img = np.zeros((32, 32))

                    x_pos = (seq[:, 0] * 32).cpu().detach().numpy().astype(np.uint16)

                    y_pos = (seq[:, 1] * 32).cpu().detach().numpy().astype(np.uint16)
                    img[y_pos, x_pos] = seq[:, 2].cpu().detach().numpy()
                    plt.imshow(img, vmin=np.min(seq[:, 2].cpu().detach().numpy()))
                    plt.colorbar()
                    plt.title(f"Prev: {i}")
                    plt.subplot(4, 3, 3 * i + 2)
                    seq = rnd_seq[i, rnd_time[i] : rnd_time[i] + 1000]
                    img = np.zeros((32, 32))

                    x_pos = (seq[:, 0] * 32).cpu().detach().numpy().astype(np.uint16)

                    y_pos = (seq[:, 1] * 32).cpu().detach().numpy().astype(np.uint16)
                    img[y_pos, x_pos] = seq[:, 2].cpu().detach().numpy()
                    plt.imshow(img, vmin=np.min(seq[:, 2].cpu().detach().numpy()))
                    plt.colorbar()
                    plt.title(f"Future: {i}")

                    plt.subplot(4, 3, 3 * i + 3)

                    st_x = rnd_seq[i, : rnd_time[i]][None, :, :]

                    x_pt = np.arange(32)
                    xx, yy = np.meshgrid(x_pt, x_pt)
                    xx = xx.reshape(-1) / 32
                    yy = yy.reshape(-1) / 32
                    img = np.zeros((32, 32))

                    back_inf = self.back_pts.unsqueeze(0).repeat(1, 1, 1)
                    for k in range(len(xx)):
                        t_ti = rnd_seq[i, rnd_time[i], -1] - torch.cat(
                            (st_x[:, :, -1], torch.zeros(1, self.cfg.num_pts).cuda()),
                            -1,
                        )

                        s_diff = torch.tensor([xx[k], yy[k]], device="cuda")[
                            None, None, :
                        ] - torch.cat((st_x[:, :, :2], back_inf), 1)

                        w_sub = torch.hstack(
                            [
                                W[rnd_plot[i], : rnd_time[i], 0],
                                W[rnd_plot[i], seq_data.shape[1] :, 0],
                            ]
                        )

                        sigmaT_sub = torch.hstack(
                            [
                                sigmaT[rnd_plot[i], : rnd_time[i], 0],
                                sigmaT[rnd_plot[i], seq_data.shape[1] :, 0],
                            ]
                        )

                        sigmaS_sub = torch.vstack(
                            [
                                sigmaS[rnd_plot[i], : rnd_time[i]],
                                sigmaS[rnd_plot[i], seq_data.shape[1] :],
                            ]
                        )

                        st = intensity(
                            w_sub[None, :],
                            sigmaT_sub[None, :],
                            t_ti,
                            s_diff,
                            sigmaS_sub[None, :, :],
                        )

                        img[
                            (32 * yy[k]).astype(np.uint16),
                            (32 * xx[k]).astype(np.uint16),
                        ] = st[0]
                    plt.imshow(np.log1p(img))
                    plt.title(f"Intensity: {i}")
                    plt.colorbar()
                global idm
                plt.savefig(f"{idm:05}.png")
                idm += 1

                self.logger.experiment.add_figure(
                    "Events",
                    fig,
                    global_step=self.trainer.global_step,
                )
                plt.close("all")

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
