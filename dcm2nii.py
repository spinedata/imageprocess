import numpy as np
import nibabel as nib
import os
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import shutil

def dcm2nii(path_read, path_save):
    reader = sitk.ImageSeriesReader()
    seriesIDs = reader.GetGDCMSeriesIDs(path_read)
    N = len(seriesIDs)
    lens = np.zeros([N])
    for i in range(N):
        dicom_names = reader.GetGDCMSeriesFileNames(path_read, seriesIDs[i])
        lens[i] = len(dicom_names)
    N_MAX = np.argmax(lens)
    dicom_names = reader.GetGDCMSeriesFileNames(path_read, seriesIDs[N_MAX])
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    sitk.WriteImage(image, path_save+'/data.nii.gz')

DICOMpath = r".\Dicomdataset"   #dicom文件夹路径
Midpath = r".\middataset"   #处理中间数据路径
Resultpath = r".\result"    #保存路径
cases = os.listdir(DICOMpath)  #获取dicom文件夹路径子文件夹名
for c in cases:   #遍历dicom文件夹路径子文件，获取多个文件
    path_mid = os.path.join(DICOMpath , c)  #获取dicom文件夹下每一套数据的路径
    dcm2nii(path_mid , Midpath )  # 将dicom转换为nii，并保存在Midpath中
    shutil.copy(os.path.join(Midpath , "data.nii.gz"), os.path.join(Resultpath , c + ".nii.gz"))


import dicom2nifti
original_dicom_directory = r".\Dicomdataset"
output_file = r'./result/test.nii'  #.nii.gz也可以
dicom2nifti.dicom_series_to_nifti(original_dicom_directory, output_file, reorient_nifti=True)
print("end")