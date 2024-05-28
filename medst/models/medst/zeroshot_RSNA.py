import datetime
import os
from argparse import ArgumentParser

import torch
from dateutil import tz
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
         
from pytorch_lightning.loggers import WandbLogger
import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer
from math import ceil, floor
from typing import Callable, List, Optional, Union
from scipy import ndimage
from MedST.medst.constants import *

from MedST.medst.datasets.classification_dataset import (CheXpertImageDataset,
                                                  COVIDXImageDataset,
                                                  RSNAImageDataset)
from MedST.medst.datasets.data_module import DataModule
from MedST.medst.datasets.transforms import DataTransforms, Moco2Transform
from MedST.medst.models.medst.medst_module import MedST
from MedST.medst.models.ssl_finetuner import SSLFineTuner
from pytorch_lightning import LightningModule
from transformers import BertTokenizer
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC 
from MedST.medst.datasets.transforms import DataTransforms
from MedST.medst.datasets.utils import get_imgs, read_from_dicom
import tempfile
from sklearn.model_selection import KFold

from pathlib import Path
import random
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class MedST_temporal(LightningModule):
    def __init__(self,
                path
                # model_name: str="resnet_50"
                ):
        super().__init__()
        # self.backbone = backbone
        self.path = path
        
        self.model = MedST.load_from_checkpoint(path, strict=False)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()   
        self.img_encoder_q = self.model.img_encoder_q
        self.text_encoder_q = self.model.text_encoder_q
        self.tokenizer = BertTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT")
        self.max_words = 112
        self.imsize = 256

    def sent_proc(self, captions):
        captions = captions.replace("\n", " ")
        # split sentences
        splitter = re.compile("[0-9]+\.")
        captions = splitter.split(captions)
        captions = [point.split(".") for point in captions]
        captions = [sent for point in captions for sent in point]
        cnt = 0
        study_sent = []
        # create tokens from captions
        for cap in captions:
            if len(cap) == 0:
                continue

            cap = cap.replace("\ufffd\ufffd", " ")
            # picks out sequences of alphanumeric characters as tokens
            # and drops everything else
            tokenizer = RegexpTokenizer(r"\w+")
            tokens = tokenizer.tokenize(cap.lower())   
            # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
            if len(tokens) <= 1:
                continue

            # filter tokens for current sentence
            included_tokens = []
            for t in tokens:
                t = t.encode("ascii", "ignore").decode("ascii")
                if len(t) > 0:
                    included_tokens.append(t)

            if len(included_tokens) > 0:
                study_sent.append(" ".join(included_tokens))
        sent = ""
        for i in study_sent:
            sent = sent + i + ""
        return sent
    
    def tokenize_input_prompts(self, prompts_input):
         
        prompts=([self.sent_proc(i) for i in prompts_input])
        tokens = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )
        return tokens

    def get_embeddings_from_prompt(self,prompt):
        tokens = self.tokenize_input_prompts(prompt)
        cap_id = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        token_type_ids = tokens["token_type_ids"]
        
        report_feat_q, word_feat_q, word_attn_q, sents = self.text_encoder_q(
                cap_id, attention_mask, token_type_ids
            )
        # word_emb_q = text_encoder_q.local_embed(word_feat_q)
        # word_emb_q = F.normalize(word_emb_q, dim=-1)
        report_emb_q = self.text_encoder_q.global_embed(report_feat_q)
        # report_emb_q1 = F.normalize(report_emb_q, dim=-1)
        return report_emb_q

    def get_pairwise_similarities(self, prompt1, prompt2):
        emb1 = self.get_embeddings_from_prompt(prompt1)
        emb2 = self.get_embeddings_from_prompt(prompt2)

        sim = torch.diag(torch.mm(emb1,emb2.t()).detach())
        return sim

    def get_image_feat(self, path_list):
        transform = DataTransforms(is_train=False)
        imgs = []
        size = []
        for i in path_list:
            # print(i)
            img = read_from_dicom(i, self.imsize, transform)
            imgs.append(img)
            # print(sizee)
        f, p = self.img_encoder_q(torch.stack(imgs))
        img_emb_q = self.img_encoder_q.global_embed(f)
        img_emb_q = F.normalize(img_emb_q, dim=-1)
        patch_emb_q = self.img_encoder_q.local_embed(p)
        patch_emb_q = F.normalize(patch_emb_q, dim=-1)
        return img_emb_q, patch_emb_q, size

    def get_similarity_score_from_raw_data(self, image_path, query_text: Union[List[str], str]) -> float:
        """Compute the cosine similarity score between an image and one or more strings.

        If multiple strings are passed, their embeddings are averaged before L2-normalization.

        :param image_path: Path to the input chest X-ray, either a DICOM or JPEG file.
        :param query_text: Input radiology text phrase.
        :return: The similarity score between the image and the text.
        """
        # assert not self.image_inference_engine.model.training
        # assert not self.text_inference_engine.model.training

        # query_text = [query_text] if isinstance(query_text, str) else query_text
        num_prompts = len(query_text)
        img_emb, _, _ = self.get_image_feat(image_path) # 1 196 128
        text_embedding = self.get_embeddings_from_prompt(query_text) # 1 128 没有norm


        assert text_embedding.shape[0] == num_prompts
        text_embedding = text_embedding.mean(dim=0)
        text_embedding = F.normalize(text_embedding, dim=0, p=2)

        cos_similarity = img_emb @ text_embedding.t()

        return cos_similarity

def _get_default_text_prompts_for_pneumonia():
    """
    Get the default text prompts for presence and absence of pneumonia
    """
    pos_query = [
        'Findings consistent with pneumonia',
        'Findings suggesting pneumonia',
        'This opacity can represent pneumonia',
        'Findings are most compatible with pneumonia',
    ]
    neg_query = [
        'There is no pneumonia',
        'No evidence of pneumonia',
        'No evidence of acute pneumonia',
        'No signs of pneumonia',
    ]
    return pos_query, neg_query

def zeroshot(path, seed):
    print(path)
    model = MedST_temporal(path)
    positive_prompts, negative_prompts = _get_default_text_prompts_for_pneumonia()

    df = pd.read_csv(RSNA_TEST_CSV)
    df = df.sample(frac=1, random_state=seed)
    df["Path"] = df["patientId"].apply(
            lambda x: RSNA_IMG_DIR / (x + ".dcm"))
    img=[]
    label=[]
    lenn = len(df)
    for i in range(lenn):
        row = df.iloc[i]
        img_path = row["Path"]
        img.append(img_path)
        y = float(row["Target"])
        y = torch.tensor([y])
        label.append(y)

    pos_socre = model.get_similarity_score_from_raw_data(image_path=img, query_text=positive_prompts)
    neg_socre = model.get_similarity_score_from_raw_data(image_path=img, query_text=negative_prompts)
    predictions = pos_socre>neg_socre

    label = torch.tensor(label)
    accuracy = accuracy_score(label, predictions)
    f1 = f1_score(label, predictions)
    auroc = roc_auc_score(label, predictions)
    print(accuracy, f1, auroc)



if __name__ == "__main__":
    # it may take a while
    zeroshot('path/to/medst.ckpt', 42)
     