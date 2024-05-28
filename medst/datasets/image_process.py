import os
from skimage import io, transform
from PIL import Image
import numpy as np
import pandas as pd
import argparse
import skimage
from tqdm import tqdm
from MedST.medst.constants import *

def get_MIMIC_img(subject_id, study_id, dicom):
    # dataset link: https://physionet.org/content/mimic-cxr/2.0.0/
    path = 'xx' # meta MIMIC path
    report_path = 'xx' # report MIMIC path

    sub_dir = 'p' + subject_id[0:2] + '/' + 'p' + subject_id + '/' + 's' + study_id + '/' + dicom + '.jpg'
    report_sub_dir = 'p' + subject_id[0:2] + '/' + 'p' + subject_id + '/' + 's' + study_id + '.txt'
    jpg_path = path + sub_dir
    report_path = report_path + report_sub_dir

    img = Image.open(jpg_path)
    img = np.array(img)
    return img

def get_and_proc(path, resize):
    img = Image.open(path)
    img = np.array(img)
    x, y = np.nonzero(img)
    xl,xr = x.min(),x.max()
    yl,yr = y.min(),y.max()
    img = img[xl:xr+1, yl:yr+1]
    img = ((img - img.min()) * (1/(img.max() - img.min()) * 256))

    img = skimage.transform.resize(img, (resize, resize), 
    order=1, preserve_range=True, anti_aliasing=False)
    img = img.astype(np.uint8)
    return img

parser = argparse.ArgumentParser(description='extract_data')
parser.add_argument('--resize', type=int)
parser.add_argument('--dataset', type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    resize = 224
    metacsv = pd.read_csv(MIMIC_CXR_ST_CSV).astype(str) # master csv from MGCA preprocessing stage

    temp_npy = np.zeros((metacsv.shape[0], resize, resize), dtype=np.uint8)
    print(metacsv.shape, temp_npy.shape)
    data = []
    iiii = 0
    with tqdm(total=len(metacsv)) as pbar:
        for row in metacsv.itertuples(index=False):
            iiii += 1
            subject_id = row[0]
            # print(row[0])
            for i in range(2,12,3): # 234 567 8910 11 12 13
                # print(row[i+2])
                if row[i + 1] == "no-value":
                    # print("2222")
                    break
                # 正视图
                path_frontal = os.path.join(subject_id[:3], subject_id ,row[i], row[i + 1]+".jpg")
                img = get_and_proc(os.path.join(MIMIC_CXR_DATA_DIR, "files", path_frontal), resize)
                data.append((path_frontal, img))
                
                if row[i + 2] == "no-value":
                    continue
                path_lateral = os.path.join(subject_id[:3], subject_id ,row[i] ,row[i + 2]+".jpg")
                img = get_and_proc(os.path.join(MIMIC_CXR_DATA_DIR, "files", path_lateral),  resize)
                data.append((path_lateral, img))
            # if iiii >= 20:
            #     break
            pbar.update(1)
        data = np.array(data, dtype=object)

    np.save(f"path2img.npy", data)
            


 
    # for i in tqdm(range(temp_npy.shape[0])):
    #     dicom_idx = metacsv['dicom_id'][i]
    #     subject_idx = str(int(metacsv['subject_id'][i]))
    #     study_idx = str(int(metacsv['study_id'][i]))
        
    #     img = get_MIMIC_img(subject_id=subject_idx, study_id=study_idx, dicom=dicom_idx)
    #     x, y = np.nonzero(img)
    #     xl,xr = x.min(),x.max()
    #     yl,yr = y.min(),y.max()
    #     img = img[xl:xr+1, yl:yr+1]
    #     img = ((img - img.min()) * (1/(img.max() - img.min()) * 256))

    #     img = skimage.transform.resize(img, (resize, resize), 
    #     order=1, preserve_range=True, anti_aliasing=False)
    #     img = img.astype(np.uint8)

    #     temp_npy[i,:,:] = img

    # np.save(f'xx', temp_npy) # save to ext_data folder