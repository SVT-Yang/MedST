# Unlocking the Power of Spatial and Temporal Information in Medical Multimodal Pre-training 【ICML 2024】

This is the offical code of 'Unlocking the Power of Spatial and Temporal Information in Medical Multimodal Pre-training'[ICML 2024]. 


### Installation:

Clone this repository and install Python dependencies:

```
git clone https://github.com/SVT-Yang/MedST.git
pip install -r requirements.txt
```

### Datasets Preparation:

Datasets we used are as follows:

* **MIMIC-CXR**:  [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) is the medical mutimodal dataset we used for pretraining.
* **MS-CXR-T benchmark**：We used [MS-CXR-T](https://physionet.org/content/ms-cxr-t/1.0.0/) benchmark for temporal downstream tasks.
* **RSNA**: We used the stage 2 of [RSNA](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data) dataset in Kaggle.

* **COVIDx**: We used the version 6 of [COVIDx](https://www.kaggle.com/datasets/andyczhao/covidx-cxr2) dataset in Kaggle which has 3 classes, i.e., no pneumonia/non-COVID-19 pneumonia/COVID-19 pneumonia.

After downloading datasets, please check if the path in `constants.py` is correct.

### Data preprocess:

run `mimic_cxr.py` to get multi-view image-text pairs and temporal information.

run `rsna.py` and `covidx.py` to get train/val/test set.

### Pre-training:

pretrained weights we used:

* Text encoder (BioClinicalBERT) : download `pytorch_model.bin` to `/medst/emilyalsentzer/Bio_ClinicalBERT` folder from [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT).

* MGCA  pre-trained weights from [MGCA](https://github.com/fuying-wang/MGCA).

Before pretraining, please make sure all the `path` are correct.

Then, we can use this command to pretrain:

```linux
cd medst/models/medst
CUDA_VISIBLE_DEVICES=0,1 python medst_module.py --gpus 2 --strategy ddp --batch_size 10  --num_workers 8
```

Our pre-trained MedST can be found [here](https://drive.google.com/file/d/1hXn7unpGYINwBGwmpiZXfYFYxVOkr-nF/view?usp=sharing).

### Downstream tasks:

First, we need set the `path` (or `ckpt_path`) argument to the path of our pre-trained [MedST](https://drive.google.com/file/d/1hXn7unpGYINwBGwmpiZXfYFYxVOkr-nF/view?usp=sharing) model.

##### 1. Temporal tasks (MS-CXR-T benchmark):

* make sure the path of two csv files (temporal image classification and temporal sentence similarity classification) are correct.
* run `temporal_test.py` to get the results.

##### 2. Zero-shot classification on RSNA:

run `zeroshot_RSNA.py` to get the results.

##### 3. Image classification on COVIDx:

 We use `--data_pct` to specify the portion of training data for finetuning. To run all experiments for COVIDx classification task, we use this command:

```
./run_cls_covidx.sh
```


### Acknowledgement

This work is built upon the [MGCA](https://github.com/fuying-wang/MGCA) and [TCC](https://github.com/June01/tcc_Temporal_Cycle_Consistency_Loss.pytorch).

### Citation

### 