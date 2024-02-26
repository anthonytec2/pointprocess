import os, sys

import hydra
import lightning as pl
import torch
from dm import PPDataModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.profiler import ProfilerActivity, profile, record_function
from pp import PPModel

torch.backends.cudnn.benchmark = True


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg) -> None:
    model = PPModel(cfg)

    # model = torch.compile(model, disable=cfg.comp)

    dm = PPDataModule(cfg)

    logger = TensorBoardLogger(
        "main",
        name=f"pp",
        version=cfg.exp,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="train_tot_epoch",
        save_top_k=3,
        every_n_epochs=1,
        # save_last=True,
        save_on_train_epoch_end=True,
        filename="epoch_{epoch}_step_{step}",
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        accelerator="cuda",
        logger=logger,
        log_every_n_steps=cfg.log_int,
        max_epochs=cfg.max_epochs,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=cfg.val_epochs,
        callbacks=[checkpoint_callback],
        precision="16-mixed",
        # limit_train_batches=0.01,
        # detect_anomaly=True
        # accumulate_grad_batches=1,
        # strategy=DDPStrategy(gradient_as_bucket_view=False, static_graph=True)
        # strategy="ddp"
        # overfit_batches=1,
        # profiler="simple",
        # max_steps=200,
    )
    if cfg.ckpt_path:
        dm.setup(stage="test")

        pred_writer = DataWriter(
            total_data=(len(dm.val_loader), len(dm.train_loader)), res=dm.res, cfg=cfg
        )
        trainer.callbacks.append(pred_writer)

        with torch.no_grad():
            trainer.predict(
                model,
                dataloaders=[dm.val_dataloader(), dm.predict_dataloader()],
                ckpt_path=cfg.ckpt_path,
                return_predictions=False,
            )

    else:
        trainer.fit(
            model=model,
            datamodule=dm,
            ckpt_path=cfg.saved_model,
        )


if __name__ == "__main__":
    run()
