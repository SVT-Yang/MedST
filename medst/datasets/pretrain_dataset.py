import os
import pickle
import re
import json
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from nltk.tokenize import RegexpTokenizer
from MedST.medst.constants import *
from MedST.medst.datasets.utils import get_imgs
from tqdm import tqdm
from transformers import BertTokenizer
from random import shuffle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class MultimodalPretrainingDataset(data.Dataset):
    def __init__(self, split="train", transform=None, data_pct=1.0,
                 imsize=256, max_words=112, sent_num=3, series_len=4):
        super().__init__()
        if not os.path.exists(MIMIC_CXR_DATA_DIR):
            raise RuntimeError(f"{MIMIC_CXR_DATA_DIR} does not exist!")

        self.transform = transform
        self.imsize = imsize
        self.series_len = series_len

        self.df = pd.read_csv(MIMIC_CXR_IT_CSV).astype(str) 
        # Ablation
        # ppp = 0.2
        # print("ablation ratio of lateral images:%f" % ppp)
        # lateral_indices = self.df['dicom_id_y'] != 'no_LT'
        # num_to_replace = int(np.sum(lateral_indices) * ppp) 
        # replace_indices = np.random.choice(np.where(lateral_indices)[0], size=num_to_replace, replace=False)  
        # self.df.loc[replace_indices, 'y'] = 'no_LT'

        if (split=="train"):
            with open(MIMIC_CXR_TEM_TRAIN_JSON, 'r') as json_file:
                self.temporal = json.loads(json.load(json_file))
        else:
            with open(MIMIC_CXR_TEM_VAL_JSON, 'r') as json_file:
                self.temporal = json.loads(json.load(json_file))

        self.study2sent = pd.read_csv(MIMIC_CXR_ALL_CSV)


        # load studies and study to text mapping
        self.filenames, self.path2sent, self.zheng2ce = self.load_text_data(split)
        
        # self.df = self.df[self.df['split'] == split]

        # self.df = self.df[self.df[MIMIC_CXR_SPLIT_COL] == split]
        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.df.reset_index(drop=True, inplace=True)
        
        self.tokenizer = BertTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT")
        self.max_words = max_words

    def load_text_data(self, split):
        # get study to captions mapping
        filepath = os.path.join(
            BASE_DIR, "../../data/captionsv1.pickle")
        filepath1 = os.path.join(
            BASE_DIR, "../../data/lateralv1.pickle")
        
        if not os.path.isfile(filepath):
            print(
                f"Caption file {filepath} does not exist. Creating captions...")
            path2sent, zheng2ce = self.create_path_2_sent_mapping()
            with open(filepath, "wb") as f:
                pickle.dump(path2sent, f, protocol=2)
                print("Save to: ", filepath)
            with open(filepath1, "wb") as f:
                pickle.dump(zheng2ce, f, protocol=2)
                print("Save to: ", filepath1)
        else:
            with open(filepath, "rb") as f:
                path2sent = pickle.load(f)
            with open(filepath1, "rb") as f:
                zheng2ce = pickle.load(f)

        # filter studies to use for current split
        filenames = []
        for row in self.df.itertuples():
            cur_split = getattr(row, MIMIC_CXR_SPLIT_COL)
            path = getattr(row, "path1")
            path2 = getattr(row, "path2")
            ce = getattr(row, "dicom_id_y")
            if cur_split == split and path in path2sent:   
                filenames.append(path)        
                if (len(ce) > 6):
                    filenames.append(path2)
    
        return filenames, path2sent, zheng2ce

    
    def create_path_2_sent_mapping(self):
        sent_lens, num_sents = [], []
        path2sent = {}
        zheng2ce = {}
        
        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            path2sent[row["path1"]] = row['content']
            if (len(row["dicom_id_y"]) > 6):
                path2sent[row["path2"]] = row['content']
                zheng2ce[row["path1"]] = row["path2"]
        return path2sent, zheng2ce

    def __len__(self):
        return len(self.temporal)

    def get_caption(self, sent):
        if len(sent) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        # series_sents = list(filter(lambda x: x != "", series_sents))
        # sent = " ".join(series_sents)

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def __getitem__(self, index):
        # temporal sequence
        series = self.temporal[index]
        
        img1, txt1, img2, txt2, ce_list = [], [], [], [], []
        caps1, caps_len1, caps2, caps_len2 = [], [], [], []
        lateral_idx = []
        imgs, txts = [], []
        caps, caps_lens = [], []

        for i in series:  # row index
            path = self.df.iloc[i]["path1"]
            txt = self.path2sent[path]  
            cap, cap_len = self.get_caption(txt)
            img = get_imgs(os.path.join(MIMIC_CXR_DATA_DIR,"files", path), self.imsize, self.transform, multiscale=False)
            if (img == "none"): continue
            ce = self.zheng2ce.get(path, "default")
            lateral_idx.append(1 if ce != "default" else 0)  

            if (ce != "default"):
                ce = get_imgs(os.path.join(MIMIC_CXR_DATA_DIR,"files", ce), self.imsize, self.transform, multiscale=False)
                if (ce == "none"): continue
                ce_list.append(ce)
            else: ce_list.append(torch.zeros(3,224,224))  # do not have the lateral view
            
            txts.append(txt)
            imgs.append(img)
            caps.append(cap)
            caps_lens.append(cap_len)

        return imgs, caps, caps_lens, ce_list, lateral_idx

def multimodal_collate_fn(batch):
    # imgs1: images in seq
    # imgs2: iamges not in seq
    imgs1, imgs2, ces, lateral_idxs  = [], [], [], []
    path = []
    ids1, tokens1, attention1 = [], [], []
    ids2, tokens2, attention2 = [], [], []
    
    for b in batch:
        imgs, caps, caps_len, ce_list, lateral_idx = b
        ces.extend(ce_list)
        if(len(imgs) == 4): 
            imgs1.extend(imgs)
            for i in range(len(caps)):
                ids1.extend(caps[i]["input_ids"])
                tokens1.extend(caps[i]["token_type_ids"])
                attention1.extend(caps[i]["attention_mask"])
        else:
            imgs2.extend(imgs)
            for i in range(len(caps)):
                ids2.extend(caps[i]["input_ids"])
                tokens2.extend(caps[i]["token_type_ids"])
                attention2.extend(caps[i]["attention_mask"])
        lateral_idxs.extend(lateral_idx)
 
    # extend
    s_len = len(imgs1)
    
    imgs1.extend(imgs2)
    ids1.extend(ids2)
    tokens1.extend(tokens2)
    attention1.extend(attention2)
    
    # stack
    imga = torch.stack(imgs1)
    idsa = torch.stack(ids1).squeeze()
    tokena = torch.stack(tokens1).squeeze()
    atta = torch.stack(attention1).squeeze()

    lateral = lateral_idxs
    ces = torch.stack(ces)

    path = np.array(path)
    return_dict = {
        "imgs": imga,
        "s_len": s_len,
        "caption_ids": idsa,
        "token_type_ids": tokena,
        "attention_mask": atta,

        "ces" : ces,
        "index_ce": lateral
    }
    return return_dict

# def multimodal_collate_fn_shuffle(batch):
#     """sort sequence"""
#     imgs1, imgs2, ces, lateral_idxs  = [], [], [], []
#     path = []
#     ids1, tokens1, attention1 = [], [], []
#     ids2, tokens2, attention2 = [], [], []
#     shuffle_label = []
    
#     for b in batch:
#         # img1, cap1, cap_len1, img2, cap2, cap_len2, ce_list, lateral_idx = b
#         imgs, caps, caps_len, ce_list, lateral_idx = b
        
#         if(len(imgs) == 4):
#             paired_data = list(zip(imgs, caps, ce_list, lateral_idx))
#             shuffle_rate = torch.rand(1).item()
#             if(shuffle_rate > 0.5): 
#                 shuffle(paired_data)
#                 imgs, caps, ce_list, lateral_idx = zip(*paired_data)
#                 shuffle_label.append(1)
#             else: shuffle_label.append(0)
#             imgs1.extend(imgs)
#             # caps1.extend(cap1)
#             # caps_len1.extend(cap_len1)
#             for i in range(len(caps)):
#                 ids1.extend(caps[i]["input_ids"])
#                 tokens1.extend(caps[i]["token_type_ids"])
#                 attention1.extend(caps[i]["attention_mask"])
#             ces.extend(ce_list)
#             lateral_idxs.extend(lateral_idx)
#         else:
#             imgs2.extend(imgs)
#             # caps2.extend(cap2)
#             # caps_len2.extend(cap_len2)
#             for i in range(len(caps)):
#                 ids2.extend(caps[i]["input_ids"])
#                 tokens2.extend(caps[i]["token_type_ids"])
#                 attention2.extend(caps[i]["attention_mask"])
#             ces.extend(ce_list)
#             lateral_idxs.extend(lateral_idx)
 
#     # extend
#     s_len = len(imgs1)
    
#     imgs1.extend(imgs2)
#     ids1.extend(ids2)
#     tokens1.extend(tokens2)
#     attention1.extend(attention2)

    
#     # stack
#     imga = torch.stack(imgs1)
#     idsa = torch.stack(ids1).squeeze()
#     tokena = torch.stack(tokens1).squeeze()
#     atta = torch.stack(attention1).squeeze()

#     # # imgs2 = torch.stack(imgs2)
#     # ids2 = torch.stack(ids2).squeeze()
#     # tokens2 = torch.stack(tokens2).squeeze()
#     # attention2 = torch.stack(attention2).squeeze()

#     # lateral = torch.stack(lateral_idxs)
#     lateral = lateral_idxs
#     ces = torch.stack(ces)

#     path = np.array(path)
#     return_dict = {
#         "imgs": imga,
#         "s_len": s_len,
#         # "imgs1": imgs1,
#         "caption_ids": idsa,
#         "token_type_ids": tokena,
#         "attention_mask": atta,

#         # "imgs2": imgs2,
#         # "caption_ids2": ids2,
#         # "token_type_ids2": tokens2,
#         # "attention_mask2": attention2,
#         "ces" : ces,
#         "index_ce": lateral,
#         "shuffle_label":shuffle_label,
#     }
#     return return_dict


if __name__ == "__main__":
    from MedST.medst.datasets.transforms import DataTransforms
    transform = DataTransforms(is_train=True)
    dataset = MultimodalPretrainingDataset(split="valid", transform=transform)
    data = dataset[0]
    # print(dataset[0])
    # print(dataset[1])
    # print(dataset[2])
    # print(dataset[3])
    # print(dataset[4])
    # print(len(dataset))  # 96011 

    pos_query = [
        'Findings consistent with pneumonia',
        'Findings suggesting pneumonia',
    ]
    print(dataset.get_caption(pos_query))