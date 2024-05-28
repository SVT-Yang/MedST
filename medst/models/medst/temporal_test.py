import datetime
import os
from argparse import ArgumentParser

import torch
from dateutil import tz
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer
from math import ceil, floor
from typing import Callable, List, Optional, Union
from scipy import ndimage
from MedST.medst.datasets.classification_dataset import COVIDXImageDataset
                                                  
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
from MedST.medst.datasets.utils import get_imgs
import tempfile
from sklearn.model_selection import KFold

from pathlib import Path
import random
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# directly test, no fine-tuning
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
            tokens = tokenizer.tokenize(cap.lower())  # 正则的tokenize
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
        report_emb_q1 = F.normalize(report_emb_q, dim=-1)
        return report_emb_q1

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
            img,sizee = get_imgs(i, self.imsize, transform, multiscale=False, return_size=True)
            imgs.append(img)
            size.append(sizee)
            # print(sizee)
        f, p = self.img_encoder_q(torch.stack(imgs))
        img_emb_q = self.img_encoder_q.global_embed(f)
        img_emb_q = F.normalize(img_emb_q, dim=-1)
        patch_emb_q = self.img_encoder_q.local_embed(p)
        patch_emb_q = F.normalize(patch_emb_q, dim=-1)
        return img_emb_q, patch_emb_q, size


    
def image_classify(path, random_seed):
    model = MedST_temporal(path)
    img_csv = "path/to/ms-cxr-t/1.0.0/MS_CXR_T_temporal_image_classification_v1.0.0.csv"
    img_df = pd.read_csv(img_csv)
    img_df = img_df.sample(frac=1, random_state=random_seed)

    consolidation = img_df[["dicom_id", "previous_dicom_id", "study_id", "subject_id", "consolidation_progression"]].dropna(subset=["consolidation_progression"])
    edema = img_df[["dicom_id", "previous_dicom_id", "study_id", "subject_id", "edema_progression"]].dropna(subset=["edema_progression"])
    pleural_effusion = img_df[["dicom_id", "previous_dicom_id", "study_id", "subject_id", "pleural_effusion_progression"]].dropna(subset=["pleural_effusion_progression"])
    pneumonia = img_df[["dicom_id", "previous_dicom_id", "study_id", "subject_id", "pneumonia_progression"]].dropna(subset=["pneumonia_progression"])
    pneumothorax = img_df[["dicom_id", "previous_dicom_id", "study_id", "subject_id", "pneumothorax_progression"]].dropna(subset=["pneumothorax_progression"])

    def classify_value(x):
        if x == "improving":
            return 0
        elif x == "stable":
            return 1
        else:
            return 2
    
    def disease_proc(disease, df, cv, seed):
        X = []
        y = df["label"].values
        PATH = "/home/yangjinxia/physionet.org/files/mimic-cxr-jpg/2.0.0/files"
        all_path = []
        for index, row in df.iterrows():
            all_path.append(os.path.join(PATH, row["dicom_id"]+".jpg"))
            all_path.append(os.path.join(PATH, row["previous_dicom_id"]+".jpg"))
            # feats = model.get_image_feat([os.path.join(PATH, row["dicom_id"]+".jpg"), os.path.join(PATH, row["previous_dicom_id"]+".jpg")]) # 2 128
        feats, _, _ = model.get_image_feat(all_path) # 402 128    201 2 128    201 256
        print(feats.shape)
        feats = feats.reshape((-1, 2, 128)).reshape((-1, 256))
        X = feats.numpy()
        classifier = SVC(kernel='linear', random_state= seed)
        scores = cross_val_score(classifier, X, y, cv=cv)
        mean_accuracy = round(scores.mean(),4)
        print(disease)
        print("Mean Accuracy:%.2f"%(mean_accuracy*100))
        return mean_accuracy*100

    consolidation["label"] = consolidation["consolidation_progression"].apply(classify_value)
    edema["label"] = edema["edema_progression"].apply(classify_value)
    pleural_effusion["label"] = pleural_effusion["pleural_effusion_progression"].apply(classify_value)
    pneumonia["label"] = pneumonia["pneumonia_progression"].apply(classify_value)
    pneumothorax["label"] = pneumothorax["pneumothorax_progression"].apply(classify_value)
    print(path, random_seed)
    cv5 = []
    cv10 = []
    for cv in [5, 10]:
        print(cv)
        avg=[]
        avg.append(disease_proc("consolidation", consolidation, cv, random_seed))
        avg.append(disease_proc("edema", edema, cv, random_seed))
        avg.append(disease_proc("pleural_effusion", pleural_effusion, cv, random_seed))
        avg.append(disease_proc("pneumonia", pneumonia, cv, random_seed))
        avg.append(disease_proc("pneumothorax", pneumothorax, cv, random_seed))
        print("avg_acc:%.2f"%(np.mean(avg)))
        avg.append(np.mean(avg))
        if cv == 5: cv5 = avg
        if cv == 10: cv10 = avg
    return cv5, cv10
    
 

def text_sim(path,seed):
    model = MedST_temporal(path)
    print(path)
    # text similarity
    txt_path = "path/to/ms-cxr-t/1.0.0/MS_CXR_T_temporal_sentence_similarity_v1.0.0.csv"
    txt_df = pd.read_csv(txt_path)
    txt_df = txt_df[txt_df['subset_name'] == 'RadGraph']
    prompt1=[]
    prompt2=[]
    label=[]
    thresh=0.5
    for row in txt_df.itertuples():
        prompt1.append(row[2])
        prompt2.append(row[3])
        if (row[4]=="contradiction"): label.append(0)
        else: label.append(1)
    
    # indices = list(range(len(prompt1)))
    # random.seed(seed)
    # random.shuffle(indices)
    # prompt1 = [prompt1[i] for i in indices]
    # prompt2 = [prompt2[i] for i in indices]
    # label = [label[i] for i in indices]

    datasize = len(label)
    label = torch.tensor(label)
    all_sim = model.get_pairwise_similarities(prompt1, prompt2)

    auc_roc = roc_auc_score(label, all_sim)
    print("overall auc:",auc_roc)

    kf = KFold(n_splits=10)
    overall_threshold = 0
    test_max_acc = 0
    for train_index, test_index in kf.split(prompt1):
        # split datasets to train/test
        train_texts1, test_texts1 = [prompt1[i] for i in train_index], [prompt1[i] for i in test_index]
        train_texts2, test_texts2 = [prompt2[i] for i in train_index], [prompt2[i] for i in test_index]
        train_labels, test_labels = [label[i] for i in train_index], [label[i] for i in test_index]
        train_labels=torch.tensor(train_labels)
        test_labels=torch.tensor(test_labels)

        # search for threshold
        sim = model.get_pairwise_similarities(train_texts1, train_texts2)
        best_threshold = 0
        best_accuracy = 0
        for threshold in np.arange(0, 1.005, 0.005):
            y_val_pred = sim > threshold
            acc = torch.sum(y_val_pred==train_labels)/len(train_texts2)
            if acc > best_accuracy:
                best_threshold=threshold
                best_accuracy=acc
        text_sim = model.get_pairwise_similarities(test_texts1, test_texts2)
        y_test_pred = text_sim>best_threshold
        test_acc = torch.sum(y_test_pred==test_labels)/len(test_labels)
        if (test_acc > test_max_acc):
            test_max_acc = test_acc
            overall_threshold = best_threshold

    all_pred = all_sim>overall_threshold
    acc = torch.sum(all_pred==label)/(datasize)
    print(seed)
    print("the threshold is: {}".format(overall_threshold))
    print("overall acc: {}".format(acc))
    
if __name__ == "__main__":
    # temporal image classification
    path = 'path_to_MedST'
    cv5 = []
    cv10 = []
    random_seeds = [42, 100, 666]
    for r in random_seeds:
        res = image_classify(path, r)
        cv5.append(res[0])
        cv10.append(res[1])
    avg_cv5 = []
    avg_cv10 = []
    std_cv5 = []
    std_cv10 = []
    
    print(path)
    print(random)
    print("avg_cv5", np.mean(cv5, axis=0))
    print("std_cv5", np.std(cv5, axis=0))
    print("avg_cv10", np.mean(cv10, axis=0))
    print("std_cv10", np.std(cv10,axis=0))
     
    # temporal sentence similarity
    text_sim(path, 42)
    
