'''
    此函数实现将nii.gz数据转换成3个维度的图片信息
'''

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from cfg import *
import nibabel as nib

import numpy as np
import pandas as pd

from PIL import Image
from skimage.io import imread
from torchvision import transforms
import torchvision.transforms.functional as TF
import cv2
from random import shuffle
from sklearn.model_selection import KFold
import torchvision.transforms as transforms
from PIL import Image
import dicom
''' 24-36行用于获取原dcm图像的切片厚度，以获得缩放比例'''
slices = {}
spacing = {}
scale_factor = {}
for s in os.listdir(dcm_path):
	print(s)
	name = str(s)
	one_patient_path = os.path.join(dcm_path,s)
	slices[name] = [dicom.read_file(os.path.join(one_patient_path, a),force=True) for a in os.listdir(one_patient_path)]
	# slices[name].sort(key=lambda x: int(x.ImagePositionPatient[2]))
	spacing[name] = map(float, ([slices[name][0].SliceThickness] + slices[name][0].PixelSpacing))
	spacing[name] = np.array(list(spacing[name]))  # [0.75  0.50195312 0.50195312]
	scale_factor[name] = spacing[name][0] / spacing[name][1]

def Hu2Gray(img):
	'''
	将Hu值图像转换为Gray图像
	img: HU值3维CT图像
	'''
	# img = nib.load(path)
	# pic = img.get_data()
	maxPixel = np.max(img)
	minPixel = np.min(img)
	imgGray = np.zeros(img.shape,dtype=np.uint8)
	dFactor = 255.0 / (maxPixel - minPixel)
	x,y,z = img.shape
	for i in range(x):
		for j in range(y):
			for k in range(z):
				imgGray[i,j,k] = int((img[i,j,k]-minPixel)*dFactor)
	imgGray[imgGray>255] = 255
	imgGray[imgGray<0] = 0
	return imgGray

def resize(img, resize_factor):
	height,width  = img.shape[0:2]   #因为转置过，所以这里的height为303  512
	print(width,height)  #512 303
	if resize_factor != 1:
		height = height * resize_factor  #height = 303 * resize_factor
		print(height)    #float
		img = cv2.resize(img, (width, int(height)))  #默认使用双线性插值
		img = cv2.flip(img, 0)  #图片垂直翻转
	return img

def CubeImg2XYZ(img,dim,name):
	'''
	获取img图像的某个方向的图像序列和名称
	img: 一个立方体
	返回：3个方向的图片
	'''
	fileList = []
	imgList = []
	x, y, z = img.shape
	print(x, y, z)  #512 512 303
	if dim == 'z':
		for k in range(z):
			_str = str(k)
			_img = img[:,:,k] # 将图像写入指定文件中
			_img = _img.transpose()
			print(_img.shape)  #512 512
			# cv2.imshow('img', _img)
			# cv2.waitKey(0)
			fileList.append(_str) #[0,1,2,3...]
			imgList.append(_img)  #imgList是图像的列表

	elif dim == 'x':
		for k in range(x):
			_str = str(k)
			_img = img[k,:,:] # 将图像写入指定文件中
			_img = _img.transpose()
			print(_img.shape)
			resize_factor = scale_factor[name]
			_img = resize(_img, resize_factor)
			# cv2.imshow('img', _img)
			# cv2.waitKey(0)
			fileList.append(_str) #[0,1,2,3...]
			imgList.append(_img)  #imgList是图像的列表
	elif dim == 'y':
		for k in range(y):
			_str = str(k)
			_img = img[:,k,:] # 将图像写入指定文件中
			_img = _img.transpose()
			print(_img.shape)
			resize_factor = scale_factor[name]
			_img = resize(_img, resize_factor)
			# cv2.imshow('img', _img)
			# cv2.waitKey(0)
			fileList.append(_str) #[0,1,2,3...]
			imgList.append(_img)  #imgList是图像的列表
	return imgList,fileList  #imgList是图像的列表， fileList是[0,1,2,3...]

def getXYZImage(pathIn, pathOut, list_):
	'''
	获取路径pathIn中所有原始图像或分割图像的某个方向的图像序列和名称
	将各方向的图像和名称输出到pathOut
	pathIn：输入图像文件夹路径
	pathOut：输出图像和名称文件路径
	z: 输出图像的维度
	x: 输出图像的维度
	返回：
	'''
	ok = False
	files = os.listdir(pathIn)  #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
	JPEGImages = []
	SegmentationImages = []
	for file in files: # 提取pathIn路径下的原始图像和分割图像文件名
		if imgType in file and strSeg not in file: # JPEGImages  imgType = '.nii'  strSeg = '_seg'
			JPEGImages.append(file)
		if imgType in file and strSeg in file: # SegmentationImages  imgType = '.nii'  strSeg = '_seg'
			SegmentationImages.append(file)
	# i=0
	for file in JPEGImages: # 输出原始图像dim维图像数据和名称  #JPEGImages装的是所有图像（不含分割）的名字   [verse004.nii.gz，verse005.nii.gz... ]
		print(file)
		img = nib.load(os.path.join(pathIn,file))  #pathIn\verse004.nii.gz
		img = img.get_data()  #获取verse004.nii.gz中的数据
		img = Hu2Gray(img)  #将Hu图像转换为灰度图像
		fname, _ = os.path.splitext(file)  # 原始文件名是xxc.nii.gz  这里fname=xxc.nii
		fname, _ = os.path.splitext(fname)  # 这里fname=xxc
		for dim in ['x', 'y', 'z']:
		# if fname in list_:   #这四句是针对verse2019数据集的
		# 	dim = 'z'
		# else:
		# 	dim = 'x'
			imgList,fileList = CubeImg2XYZ(img, dim, fname)   #上一步的img是三维图像  要获得Z轴的图像， imgList是图像的列表， fileList是[0,1,2,3...]
			strExpand = '.jpeg'
			if dim == 'x':
				_pathOut = os.path.join(pathOut, 'image', 'imagex')
			elif dim == 'y':
				_pathOut = os.path.join(pathOut, 'image', 'imagey')
			else:
				_pathOut = os.path.join(pathOut, 'image', 'imagez')
			# os.chdir(_pathOut)  #os.chdir() 方法用于改变当前工作目录到指定的路径
			for (_img, _file) in zip(imgList, fileList):
				_str1 = fname + '_' + _file  # 这里fname=xxc    xxc_0
				_str2 = _str1 + strExpand  #xxc_0.jpeg
				ok1 = cv2.imwrite(os.path.join(_pathOut, _str2), _img)

	for file in SegmentationImages: # 输出分割图像dim维图像数据和名称
		print(file)
		img = nib.load(os.path.join(pathIn,file))
		img = img.get_data() #获取verse004_seg.nii.gz中的数据
		fname, _ = os.path.splitext(file)  # 原始文件名是xxc.nii.gz  这里fname=xxc.nii
		# fname, _ = os.path.splitext(fname)  # 这里fname=xxc
		fname = fname[0:8]
		for dim in ['x', 'y', 'z']:
			# if fname in list_:
			# 	dim = 'z'
			# else:
			# 	dim = 'x'
			imgList, fileList = CubeImg2XYZ(img, dim, fname)
			strExpand = '.png'
			if dim == 'x':
				_pathOut = os.path.join(pathOut, 'segimage', 'segimagex')
			elif dim == 'y':
				_pathOut = os.path.join(pathOut, 'segimage', 'segimagey')
			else:
				_pathOut = os.path.join(pathOut, 'segimage', 'segimagez')
			# os.chdir(_pathOut)
			for (_img, _file) in zip(imgList, fileList):
				_str1 = fname + '_' + _file  #这里fname=xxc    xxc_0
				_str2 = _str1 + strExpand   #加后缀，  xxc_0.png
				ok1 = cv2.imwrite(os.path.join(_pathOut, _str2), _img)
	ok = True
	return ok

if __name__ == '__main__':

	#此list_是针对公共数据集verse2019而设置的，因为在原nii数据中，有的是矢状面位于x，有的位于z
	# list_ = ['verse022', 'verse071', 'verse073', 'verse078', 'verse080', 'verse093', 'verse116', 'verse124', 'verse125',
	# 		 'verse150', 'verse221', 'verse225', 'verse230', 'verse235', 'verse264', 'verse269', 'verse290']
	list_ = []
	imgType = '.nii'
	strSeg = '_seg'
	ok = getXYZImage(pathVerse2019, pathOut, list_)
	print('end')


