# EpiS_LFSR
This repository contains official pytorch implementation of Learning Epipolar-Spatial Relationship for Light Field Image Super-Resolution, by Ahmed Salem, Hatem Ibrahem, and Hyun-Soo Kang.

# Requirements:
PyTorch 1.13.1, torchvision 0.14.1. The code is tested with python=3.10, cuda=11.7.\
Matlab (For training/test data generation and performance evaluation)

# Datasets:
We used the EPFL, HCInew, HCIold, INRIA and STFgantry datasets for training and test. Please first download our dataset via [this link](https://stuxidianeducn-my.sharepoint.com/personal/zyliang_stu_xidian_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fzyliang%5Fstu%5Fxidian%5Fedu%5Fcn%2FDocuments%2Fdatasets&ga=1), and place the 5 datasets to the folder ./Datasets/.

# Train:
Run Generate_Data_for_Train.m to generate training data. The generated data will be saved in ./Data/train_kxSR_AxA/.\
Run train.py to perform network training.\
Checkpoint will be saved to ./log/.

# Test:
Run Generate_Data_for_Test.m to generate test data.\
Run test_on_dataset.py to perform test on each dataset.\
The original result files and the metric scores will be saved to ./Results/.

# Acknowledgement
Our work and implementations are based on the following project\
[DistgSSR](https://github.com/YingqianWang/DistgSSR)\
We sincerely thank the authors for sharing their code and amazing research work!
