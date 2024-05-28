import sys
import numpy as np
import re
import pandas as pd
from MedST.medst.constants import *
from nltk.tokenize import RegexpTokenizer
from MedST.medst.preprocess.utils import extract_mimic_text
from pandas import Series, DataFrame
from tqdm import tqdm
import json
 
extract_text = True
np.random.seed(42)

def content_combine():
    img = []
    txxxt = []
    text_df = pd.read_csv(MIMIC_CXR_TEXT_CSV)
    # print(len(text_df))
    # text_df.dropna(subset=["impression", "findings"], how="all", inplace=True)
    # print(len(text_df))
    text_df = text_df[["study", "impression", "findings", "last_paragraph", "comparison"]].astype(str)
    text_df.rename(columns={"study": "study_id"}, inplace=True)

    txt = ""
    for row in text_df.itertuples(index=False):
        name = row[0]
        if (len(row[1].split()) >= 3):  # impression
            txt += row[1]
        txt += " "
        if (len(row[2].split()) > 3):  # findings
            txt += row[2]
            txt += " "
        if (len(row[3].split()) > 6):  # last_para
            txt += row[3]
            txt += " "
        if (len(row[4].split()) > 6):  # comparison
            txt += row[4]

        if(len(txt.strip()) == 0): continue
        if(len(txt.split()) < 3): continue

        img.append(name)
        txxxt.append(txt)
        txt = ""
    datttt = {
        "study_id" : img,
        "content" : txxxt
    }
    frame = DataFrame(datttt)
    frame.to_csv(MIMIC_CXR_ALL_CSV, index=None)

def main():
    

    metadata_df = pd.read_csv(MIMIC_CXR_META_CSV, engine='python')
    metadata_df = metadata_df[["dicom_id", "subject_id",
                               "study_id", "ViewPosition", "StudyDate", "StudyTime"]]   #.astype(str)
    metadata_df["study_id"] = (metadata_df["study_id"].astype(str)).apply(lambda x: "s"+x) 
    metadata_df["subject_id"] = (metadata_df["subject_id"].astype(str)).apply(lambda x: "p"+x) 

    # frontal and lateral images
    metadata_df_fron = metadata_df[metadata_df["ViewPosition"].isin(["PA", "AP"])]
    metadata_df_la = metadata_df[metadata_df["ViewPosition"].isin(["LL", "LATERAL"])]
    metadata_df_la = metadata_df_la[["dicom_id","study_id","subject_id"]]
   
    # text_df = pd.read_csv(MIMIC_CXR_ALL_CSV)
    text_df = pd.read_csv(MIMIC_CXR_TEXT_CSV)
    text_df.dropna(subset=["impression", "findings"], how="all", inplace=True)
    text_df = text_df[["study", "impression", "findings"]]
    text_df.rename(columns={"study": "study_id"}, inplace=True)

    # split 
    split_df = pd.read_csv(MIMIC_CXR_SPLIT_CSV).astype(str)
    split_df["study_id"] = split_df["study_id"].apply(lambda x: "s"+x)
    split_df["subject_id"] = split_df["subject_id"].apply(lambda x: "p"+x)
    # TODO: merge validate and test into test.
    # split_df["split"] = split_df["split"].apply(lambda x: "valid" if x == "validate" or x == "test" else x)
    split_df["split"] = split_df["split"].apply(lambda x: "valid" if x == "validate" else x)

    # 加入文本、split列
    # meta:胸片id 对应subject-id  病人study-id  text: study imp.. find..
    master_df = pd.merge(metadata_df_fron, split_df, on=["dicom_id", "subject_id", "study_id"], how="left") # 这三个唯一标识一个胸片
    master_df = pd.merge(master_df, metadata_df_la, on=["study_id","subject_id"], how="left")
    master_df = pd.merge(master_df, text_df, on="study_id", how="left") # 通过study-id左连接一下
    print(len(master_df))
    master_df = master_df.drop_duplicates(subset=['dicom_id_x'])
    print(len(master_df))
    master_df.dropna(subset=['impression', 'findings'],how="all", inplace=True)
    print(len(master_df))
    master_df[["impression"]] = master_df[["impression"]].fillna(" ")
    master_df[["findings"]] = master_df[["findings"]].fillna(" ")
    master_df[["dicom_id_y"]] = master_df[["dicom_id_y"]].fillna("no_LT")



    # content
    content = []
    drop_flag = []
    for index, row in tqdm(master_df.iterrows(), total=master_df.shape[0]):
        captions = ""
        captions += str(row["impression"])
        captions += " "
        captions += str(row["findings"])
        # use space instead of newline
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

            cnt += len(included_tokens)

        if cnt >= 3:
            # sent_lens.append(cnt)
            # num_sents.append(len(study_sent))
            # path2sent[row[MIMIC_CXR_PATH_COL]] = study_sent
            # master_df.at[index,'content']=study_sent
            series_sents = list(filter(lambda x: x != "", study_sent))
            sent = " ".join(series_sents)
            content.append(sent)
            drop_flag.append(0)
        else:
            # master_df.drop(master_df.index[index], inplace=True)
            # print(len(master_df))
            drop_flag.append(1)
    print(len(content))
    # print(master_df)
    master_df["drop_flag"] = drop_flag
    master_df = master_df[~(master_df['drop_flag'] == 1)]
    print(len(master_df))
    master_df['content'] = content
    # master_df = df_new.reset_index()
    
    master_df.drop('drop_flag', axis=1, inplace=True)
    master_df.drop('impression', axis=1, inplace=True)
    master_df.drop('findings', axis=1, inplace=True)
    print(master_df.keys())
    # path
    n = len(master_df)
    master_data = master_df.values
    path = []
    path1 = []
    for i in range(n):
        row = master_data[i]
        file_path = "%s/%s/%s/%s.jpg" % (str(row[1])[:3], str(row[1]), str(row[2]), str(row[0]))
        path.append(file_path)
        # print(path)
        if (pd.isna(row[7])):
            path1.append(0)
        else:
            file_path1 = "%s/%s/%s/%s.jpg" % (str(row[1])[:3], str(row[1]),str(row[2]), str(row[7]))
            path1.append(file_path1)
            # print(path1)
        # exit()
    master_df.insert(loc=0, column="path1", value=path)
    master_df.insert(loc=9, column="path2", value=path1)
    master_df.to_csv(MIMIC_CXR_IT_CSV, index=False)

def temporal_modeling(split, series_len = 4):
    df_all = pd.read_csv(MIMIC_CXR_IT_CSV)
    df_all["row_number"] = range(len(df_all))
    # train/val
    df = df_all[df_all["split"]==split]
    
    df = df.groupby("subject_id")

    # Sort each group by time, then split them into segments with a maximum length of len.
    sorted_groups = []
    item = []
    max_num = series_len
    frames = {}
    global_len = 0
    combined = pd.DataFrame()
    my_dict = {key: 0 for key in range(7)}
    ggg = 0
    for _, group in df:
        aa = group.sort_values(by=["StudyDate", "StudyTime"], ascending=[True, True])
        num_sub = len(aa) // max_num  
        left = len(aa) % max_num
        s = 0
        e = 0
        for i in range(num_sub): 
            s = i * max_num
            e = (i + 1) * max_num
            # frames[global_len] = aa.iloc[s : e].to_json(orient="records") # 全局索引：一个df
            frame = aa.iloc[s : e]
            # 直接存地址
            sorted_groups.append(frame["row_number"].to_list())
            my_dict[len(frame)] += 1

        if left != 0:
            frame = aa.iloc[e:]
            sorted_groups.append(frame["row_number"].to_list())
            my_dict[len(frame)] += 1
    print(len(sorted_groups))
    json_data = json.dumps(sorted_groups)
    if (split=="train"):
        with open(os.path.join(MIMIC_CXR_DATA_DIR,'temporal_train_len{}.json'.format(series_len)), 'w') as json_file:
            json.dump(json_data, json_file)
    else: 
        with open(os.path.join(MIMIC_CXR_DATA_DIR,'temporal_val_len{}.json'.format(series_len)), 'w') as json_file:
            json.dump(json_data, json_file)

if __name__ == "__main__":
    if extract_text:
        extract_mimic_text()
    content_combine()
    main()
    temporal_modeling("train", 4)
    temporal_modeling("valid", 4)


#     with open(MIMIC_CXR_TEM_JSON, 'r') as json_file:
#         temporal = json.load(json_file)
#     print(type(json.loads(temporal)))
#     temporal = json.loads(temporal)
#     # temporal = eval(temporal)
#     csv = pd.read_csv('./it.csv')
#     for i in temporal[0]:
#         print(i)
#         print(csv.iloc[i])