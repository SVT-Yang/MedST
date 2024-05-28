import datetime
import os
from argparse import ArgumentParser

import torch
from dateutil import tz
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from MedST.medst.datasets.classification_dataset import (CheXpertImageDataset,
                                                  COVIDXImageDataset,
                                                  RSNAImageDataset)
from MedST.medst.datasets.data_module import DataModule
from MedST.medst.datasets.transforms import DataTransforms, Moco2Transform
from MedST.medst.models.medst.medst_module import MedST
from MedST.medst.models.ssl_finetuner import SSLFineTuner

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="covidx")
    parser.add_argument("--path", type=str,
                        default="PATH_TO_MEDST")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data_pct", type=float, default=0.01)
    # add trainer args
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    print(args.path)
    # set max epochs
    args.max_epochs = 200

    seed_everything(args.seed)

    # if args.dataset == "chexpert":
    #     # define datamodule
    #     # check transform here
    #     datamodule = DataModule(CheXpertImageDataset, None,
    #                             Moco2Transform, args.data_pct,
    #                             args.batch_size, args.num_workers)
    #     num_classes = 5
    #     multilabel = True
    # elif args.dataset == "rsna":
    #     datamodule = DataModule(RSNAImageDataset, None,
    #                             DataTransforms, args.data_pct,
    #                             args.batch_size, args.num_workers)
    #     num_classes = 1
    #     multilabel = True
    if args.dataset == "covidx":
        datamodule = DataModule(COVIDXImageDataset, None,
                                DataTransforms, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 3
        multilabel = False
    else:
        raise RuntimeError(f"no dataset called {args.dataset}")

    if args.path:
        model = MedST.load_from_checkpoint(args.path, strict=False)
    else:
        model = MedST()

    args.model_name = model.hparams.img_encoder
    args.backbone = model.img_encoder_q
    # args.temporal_embed = model.temporal_embed
    args.in_features = args.backbone.feature_dim
    args.num_classes = num_classes
    args.multilabel = multilabel

    # finetune
    tuner = SSLFineTuner(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../../data/ckpts/medst_finetune/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=1),
        EarlyStopping(monitor="val_loss", min_delta=0.,
                      patience=10, verbose=False, mode="min")
    ]

    # get current time
    now = datetime.datetime.now(tz.tzlocal())

    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    logger_dir = os.path.join(
        BASE_DIR, f"../../../data/wandb")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="medst_finetune",
        save_dir=logger_dir,
        name=f"{args.dataset}_{args.data_pct}_{extension}")
    trainer = Trainer.from_argparse_args(
        args,
        deterministic=True,
        callbacks=callbacks,
        logger=wandb_logger)

    tuner.training_steps = tuner.num_training_steps(trainer, datamodule)

    # train
    trainer.fit(tuner, datamodule)
    # test
    trainer.test(tuner, datamodule, ckpt_path="best")


if __name__ == "__main__":
    cli_main()
