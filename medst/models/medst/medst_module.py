import datetime
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from MedST.medst.utils.cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from dateutil import tz
from einops import rearrange
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDP2Plugin, DDPPlugin
from MedST.medst.datasets.data_module import DataModule
from MedST.medst.datasets.pretrain_dataset import (MultimodalPretrainingDataset,
                                            multimodal_collate_fn)
from MedST.medst.datasets.transforms import DataTransforms
from MedST.medst.models.backbones.encoder import BertEncoder, ImageEncoder
from MedST.medst.models.backbones.vits import Mlp

from torch import distributed as dist
from MedST.medst.models.medst.temporal_alignment import compute_deterministic_alignment_loss
from timm.models.layers import trunc_normal_, DropPath


torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class MedST(LightningModule):
    '''Pytorch lightning implementation of MedST'''

    def __init__(self,
                 img_encoder: str = "vit_base",
                 freeze_bert: bool = False,
                 emb_dim: int = 128,
                 softmax_temperature: float = 0.07,
                 learning_rate: float = 2e-5,
                 momentum: float = 0.9,
                 weight_decay: float = 0.05,
                 batch_size: int = 64,
                 num_workers: int = 8,
                 local_temperature: float = 0.1,
                 bidirectional: bool = True,
                 use_local_atten: bool = False,
                 num_heads: int = 1,
                 lamb: float = 0.75,
                 lambda_1: float = 1,
                 lambda_2: float = 1.0,
                 lambda_3: float = 1.0,
                 epsilon: float = 0.05,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        # init encoders
        self.img_encoder_q = ImageEncoder(
            model_name=img_encoder, output_dim=self.hparams.emb_dim)
        self.text_encoder_q = BertEncoder(
            output_dim=self.hparams.emb_dim, freeze_bert=freeze_bert)
        # patch local attention layer
        self.patch_local_atten_layer = nn.MultiheadAttention(
            self.hparams.emb_dim, self.hparams.num_heads, batch_first=True)
        # sentence local attention layer
        self.word_local_atten_layer = nn.MultiheadAttention(
            self.hparams.emb_dim, self.hparams.num_heads, batch_first=True)
        
        self.word_element_wise_multi_head = nn.MultiheadAttention(
            self.hparams.emb_dim, self.hparams.num_heads, batch_first=True)
        
        self.patch_element_wise_multi_head = nn.MultiheadAttention(
            self.hparams.emb_dim, self.hparams.num_heads, batch_first=True)
       
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def comput_alignment_loss(self, embs1, embs2, num_steps=4,  loss_type="regression_mse_var", similarity_type="l2", temperature=0.1, variance_lambda = 0.001,huber_delta=None):
        loss = compute_deterministic_alignment_loss(embs1,
            embs2,num_steps,loss_type,similarity_type,temperature,variance_lambda,huber_delta)

        return loss

    def forward(self, batch, batch_idx, split="train"):
        '''Forward step of our method'''
        imgs = batch["imgs"]
        cap_id = batch["caption_ids"]
        token_type_ids = batch["token_type_ids"]
        attention_mask = batch["attention_mask"]

        s_len = batch["s_len"]
        

        # Forward of query text encoder
        report_feat_q, word_feat_q, word_attn_q, sents = self.text_encoder_q(
            cap_id, attention_mask, token_type_ids,series_len=s_len
        )
        word_emb_q = self.text_encoder_q.local_embed(word_feat_q)
        word_emb_q = F.normalize(word_emb_q, dim=-1)
        report_emb_q = self.text_encoder_q.global_embed(report_feat_q)
        report_emb_q = F.normalize(report_emb_q, dim=-1)

        indexx = torch.tensor(batch["index_ce"])
        ces = batch["ces"]

        img_feat_q, patch_feat_q = self.img_encoder_q(
            (imgs, ces, (indexx+1)%2), view_type="frontal", series_len=s_len)
        patch_emb_q = self.img_encoder_q.local_embed(patch_feat_q)
        patch_emb_q = F.normalize(patch_emb_q, dim=-1)
        img_emb_q = self.img_encoder_q.global_embed(img_feat_q)
        img_emb_q = F.normalize(img_emb_q, dim=-1)
        
        # image-text contrastive learning
        bz = img_emb_q.size(0)
        labels = torch.arange(bz).type_as(report_emb_q).long()

        scores = img_emb_q.mm(report_emb_q.t())
        scores /= self.hparams.softmax_temperature
        scores1 = scores.transpose(0, 1)
        loss0 = F.cross_entropy(scores, labels)
        loss1 = F.cross_entropy(scores1, labels)

        loss_ita = loss0 + loss1

        # # compute retrieval accuracy

        i2t_acc1, i2t_acc5 = self.precision_at_k(
            scores, labels, top_k=(1, 5))
        t2i_acc1, t2i_acc5 = self.precision_at_k(
            scores1, labels, top_k=(1, 5))
        acc1 = (i2t_acc1 + t2i_acc1) / 2.
        acc5 = (i2t_acc5 + t2i_acc5) / 2.


        ################  temporal_modeling  ########
        if (int(s_len/4) == 0): loss_temporal = torch.tensor([0.0], requires_grad=True).to(loss_ita.device)
        else:
            s_shape = (int(s_len/4), 4, -1)
            loss_series1 = self.comput_alignment_loss(img_emb_q[:s_len, :].reshape(s_shape), report_emb_q[:s_len, :].reshape(s_shape),huber_delta=self.hparams.huber_delta)
            loss_series2 = self.comput_alignment_loss(report_emb_q[:s_len, :].reshape(s_shape), img_emb_q[:s_len, :].reshape(s_shape),huber_delta=self.hparams.huber_delta)

            loss_temporal = loss_series1 + loss_series2


        ########## Modality-Weighted Local Alignment ################
        # cross attention patch to sentences
        # 从pad开始 true
        mask = torch.from_numpy(np.array(sents)[:, 1:] == "[PAD]").type_as(
            batch["imgs"]).bool()

        if self.hparams.use_local_atten:
            word_atten_output, _ = self.word_local_atten_layer(
                word_emb_q, patch_emb_q, patch_emb_q)
        else:
            atten_sim = torch.bmm(word_emb_q, patch_emb_q.permute(0, 2, 1))
            word_num = word_emb_q.size(1) 
            atten_scores = F.softmax(
                atten_sim / self.hparams.local_temperature, dim=-1)  
            word_atten_output = torch.bmm(atten_scores, patch_emb_q)
        # k_i^j
        word_atten_output = F.normalize(word_atten_output, dim=-1)

        # word_atten_output和word_embed 逐元素相乘
        a_i = torch.mul(word_emb_q, word_atten_output)  # bs, 111, 128
        a_i = F.normalize(a_i, dim=-1)
        mean_query = torch.mean(a_i, dim=1) # bs,128
        new_atten, weight_scores = self.word_element_wise_multi_head(mean_query.unsqueeze(1), a_i, a_i)
        word_atten_weights = weight_scores.squeeze()

        word_atten_weights /= word_atten_weights.sum(dim=1, keepdims=True)
        word_sim = torch.bmm(word_emb_q, word_atten_output.permute(
            0, 2, 1)) / self.hparams.local_temperature
        # print(word_sim.shape) # 144 111 111
        word_num = word_sim.size(1)
        word_sim_1 = rearrange(word_sim, "b n1 n2 -> (b n1) n2") # word_num*bz, word_num
        targets = torch.arange(word_num).type_as(
            word_emb_q).long().repeat(bz)  # word_num*bz [0,1,2,...110, 0,1,2,...,110,...]
        loss_word_1 = torch.sum(F.cross_entropy(
            word_sim_1, targets, reduction="none") * word_atten_weights.view(-1)) / bz

        word_sim_2 = rearrange(word_sim, "b n1 n2 -> (b n2) n1")
        loss_word_2 = torch.sum(F.cross_entropy(
            word_sim_2, targets, reduction="none") * word_atten_weights.view(-1)) / bz

        loss_word = (loss_word_1 + loss_word_2) / 2.

        if self.hparams.bidirectional:
            if self.hparams.use_local_atten: 
                patch_atten_output, _ = self.patch_local_atten_layer(
                    patch_emb_q, word_emb_q, word_emb_q, key_padding_mask=mask)
            else:
                atten_sim = torch.bmm(patch_emb_q, word_emb_q.permute(0, 2, 1))
                patch_num = patch_emb_q.size(1)
                atten_sim[mask.unsqueeze(1).repeat(
                    1, patch_num, 1)] = float("-inf")
                atten_scores = F.softmax(
                    atten_sim / self.hparams.local_temperature, dim=-1)
                patch_atten_output = torch.bmm(atten_scores, word_emb_q)

            # word_atten_output和word_embed 逐元素相乘
            b_i = torch.mul(patch_emb_q, patch_atten_output)  # bs, 111, 128
            b_i = F.normalize(b_i, dim=-1)
            mean_queryb = torch.mean(b_i, dim=1) # bs,128
            new_atten, weight_scores = self.patch_element_wise_multi_head(mean_queryb.unsqueeze(1), b_i, b_i)
            patch_atten_weights = weight_scores.squeeze()

            patch_sim = torch.bmm(patch_emb_q, patch_atten_output.permute(
                0, 2, 1)) / self.hparams.local_temperature
            patch_num = patch_sim.size(1)
            patch_sim_1 = rearrange(patch_sim, "b n1 n2 -> (b n1) n2")
            targets = torch.arange(patch_num).type_as(
                patch_emb_q).long().repeat(bz)
            loss_patch_1 = torch.sum(F.cross_entropy(
                patch_sim_1, targets, reduction="none") * patch_atten_weights.view(-1)) / bz

            patch_sim_2 = rearrange(patch_sim, "b n1 n2 -> (b n2) n1")
            loss_patch_2 = torch.sum(F.cross_entropy(
                patch_sim_2, targets, reduction="none") * patch_atten_weights.view(-1)) / bz

            loss_patch = (loss_patch_1 + loss_patch_2) / 2.

            loss_local = loss_patch + loss_word

        else:
            loss_local = loss_word

        return loss_ita, loss_local, loss_temporal, acc1, acc5
        

    def training_step(self, batch, batch_idx):
        loss_ita, loss_local, loss_temporal, acc1, acc5 = self(
            batch, batch_idx, "train")
        loss = self.hparams.lambda_1 * loss_ita + self.hparams.lambda_2 * \
            loss_local + self.hparams.lambda_3 * loss_temporal

        log = {
            "train_loss": loss,
            "train_loss_ita": self.hparams.lambda_1 * loss_ita,
            "train_loss_local": self.hparams.lambda_2 * loss_local,
            "train_loss_temporal": self.hparams.lambda_3 * loss_temporal,
            "train_acc1": acc1,
            "train_acc5": acc5
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):   
        loss_ita, loss_local, loss_temporal, acc1, acc5  = self(
            batch, batch_idx, "valid")

        loss = self.hparams.lambda_1 * loss_ita  + self.hparams.lambda_2 * \
            loss_local + self.hparams.lambda_3 * loss_temporal
        log = {
            "val_loss": loss,
            "val_loss_ita": self.hparams.lambda_1 * loss_ita,
            "val_loss_local": self.hparams.lambda_2 * loss_local,
            "val_loss_temporal": self.hparams.lambda_3 * loss_temporal,
            "val_acc1": acc1,
            "val_acc5": acc5
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)
        return loss
    

    @staticmethod
    def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            self.hparams.learning_rate,
            betas=(self.hparams.momentum, 0.999),
            weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.training_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=1e-8,
            warmup_steps=int(self.training_steps * 0.4)
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--img_encoder", type=str, default="vit_base")
        parser.add_argument("--freeze_bert", action="store_true")
        parser.add_argument("--emb_dim", type=int,
                            default=128, help="128, 256")
        parser.add_argument("--num_workers", type=int, default=16)
        parser.add_argument("--softmax_temperature", type=float, default=0.07)
        parser.add_argument("--learning_rate", type=float, default=4e-5)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=0.05)
        parser.add_argument("--batch_size", type=int, default=144)
        parser.add_argument("--num_heads", type=int, default=1)
        parser.add_argument("--experiment_name", type=str, default="")
        parser.add_argument("--lambda_1", type=float, default=1.)
        parser.add_argument("--lambda_2", type=float, default=1.)
        parser.add_argument("--lambda_3", type=float, default=1.)
        parser.add_argument("--huber_delta", type=float, default=2.)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--bidirectional", action="store_false") # 默认值是true
        # parser.add_argument("--use_local_atten", action="store_false")
        parser.add_argument("--data_pct", type=float, default=1.)
        parser.add_argument("--ckpt_path", type=str,
                        default="path/to/checkpoint")
        return parser

    @staticmethod
    def _use_ddp_or_dpp2(trainer: Trainer) -> bool:
        if trainer:
            return isinstance(trainer.training_type_plugin, (DDPPlugin, DDP2Plugin))
        else:
            return torch.distributed.is_initialized()

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_gpus, trainer.num_processes)
        if trainer.tpu_cores:
            num_devices = max(num_devices, trainer.tpu_cores)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices

        return (dataset_size // effective_batch_size) * trainer.max_epochs


@torch.no_grad()
def concat_all_gather(tensor):
    '''
    Performs all_gather operation on the provided tensors
    '''
    tensors_gather = [torch.ones_like(tensor) for _ in range(
        torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def cli_main():
    parser = ArgumentParser()
    # trainer args
    parser = Trainer.add_argparse_args(parser)
    # model args
    parser = MedST.add_model_specific_args(parser)
    args = parser.parse_args()

    args.deterministic = True
    args.max_epochs = 8

    # seed
    seed_everything(args.seed)

    datamodule = DataModule(MultimodalPretrainingDataset, multimodal_collate_fn,
                            DataTransforms, args.data_pct,
                            args.batch_size, args.num_workers)

    # Add load from checkpoint
    model = MedST(**args.__dict__)
    new_model_state_dict = model.state_dict()
    # old_model_state_dict = torch.load("path/to/MGCA/vit_base.ckpt", map_location="cpu")
    old_model_state_dict = torch.load("/home/yangjinxia/MGCA/data/ckpts/vit_base.ckpt", map_location="cpu")
    old_model_state_dict = {k: v for k, v in old_model_state_dict["state_dict"].items() if k in new_model_state_dict}
    model.load_state_dict(old_model_state_dict, strict=False)
    
    # for name,param in model.named_parameters():
    #     param.requires_grad = False
    #     if 'mlp_l' in name or 'norm2_l' in name :
    #         param.requires_grad = True
    #         print(name)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../../data/ckpts/MedST/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=5),
        EarlyStopping(monitor="val_loss", min_delta=0.,
                      patience=5, verbose=False, mode="min")
    ]
    logger_dir = os.path.join(
        BASE_DIR, f"../../../data")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="MedST", save_dir=logger_dir, name=extension)
    trainer = Trainer.from_argparse_args(
        args=args,
        callbacks=callbacks,
        logger=wandb_logger)

    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(model.training_steps)
    # trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path)
    trainer.fit(model, datamodule=datamodule)

    best_ckpt_path = os.path.join(ckpt_dir, "best_ckpts.yaml")
    callbacks[1].to_yaml(filepath=best_ckpt_path)


if __name__ == "__main__":
    cli_main()
