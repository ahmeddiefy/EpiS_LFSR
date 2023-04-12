# EpiS_LFSR
This repository contains official pytorch implementation of Learning Epipolar-Spatial Relationship for Light Field Image Super-Resolution, by Ahmed Salem, Hatem Ibrahem, and Hyun-Soo Kang.

# Requirement
PyTorch 1.13.1, torchvision 0.14.1. The code is tested with python=3.10, cuda=11.7.\
Matlab (For training/test data generation and performance evaluation)

# Datasets:
The datasets used in our paper can be downloaded through [this link](https://stuxidianeducn-my.sharepoint.com/personal/zyliang_stu_xidian_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fzyliang%5Fstu%5Fxidian%5Fedu%5Fcn%2FDocuments%2FLFASR%2Fdatasets&ga=1).

# Train:
Run Generate_Data_for_Training_2x2-7x7.m to generate training data.\
Run train.py to perform network training.\
Checkpoint will be saved to ./log/.

# Test:
Run Generate_Data_for_Test.m to generate test data.\
Run test.py to perform network inference.\
The PSNR and SSIM values of each dataset will be saved to ./log/.

# Acknowledgement
Our work and implementations are based on the following project\
[DistgASR](https://github.com/YingqianWang/DistgASR)\
We sincerely thank the authors for sharing their code and amazing research work!
