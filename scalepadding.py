'''
    本函数实现图片的padding、扩充预处理

'''
import cv2
import math
import os
import random
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
from  torchvision import utils as vutils

def scalepadding(img, label):
    height, width= img.shape[0:2]
    if width > height:
        pad = width - height  #如果宽大于高
        upad, dpad = math.ceil(pad / 2), math.floor(pad / 2)
        lpad, rpad = 0, 0
    elif width < height:
        pad = height - width
        lpad, rpad = math.ceil(pad / 2), math.floor(pad / 2)
        upad, dpad = 0, 0
    else:
        upad, dpad, lpad, rpad= 0, 0, 0, 0
    img = cv2.copyMakeBorder(img, upad, dpad, lpad, rpad, cv2.BORDER_CONSTANT, value=0)
    label = cv2.copyMakeBorder(label, upad, dpad, lpad, rpad, cv2.BORDER_CONSTANT, value=0)
    return img[:, :, 0], label[:, :, 0]

def GetMeanStd(path):
	# 计算训练集中图像的均值和方差
	# 输入为RGB图像
    files = os.listdir(path)
    _mean = np.array((0.0,0.0,0.0))
    _std = np.array((0.0,0.0,0.0))
    for file in files:
        img = cv2.imread(os.path.join(path, file))
        img = np.array(img)
        # img = img/255.0
        _mean = _mean + np.array((img[:,:,0].mean(),img[:,:,1].mean(),img[:,:,2].mean()))
        _std = _std + np.array((img[:,:,0].std(),img[:,:,1].std(),img[:,:,2].std()))
        _mean = _mean/len(files)
        _std = _std/len(files)
    return _mean, _std

def img_augment(data, label,augCode):  #需要PIL image
	# 图像增强
	# 全部图像均进行随机剪裁，high<=highSpine,width<=widthSpine，然后缩放到high=highSpine,width=widthSpine
	'''
	augCode:    0，无增强
				1，水平翻转
				2，垂直翻转
				3，旋转45度
				4，旋转-45度
				5，亮度50%~150%随机值
				6，对比度50%~150%随机值
	'''
	if augCode not in [0, 1, 2, 3, 4, 5, 6]:
		print("数据增强码输入错误，请输入小于6的正整数")
		return
	# data, label = rand_crop(data, label)  # 随机剪裁图像
	if augCode == 1:  # 水平翻转
		data = TF.hflip(img=data)
		label = TF.hflip(img=label)
	if augCode == 2:  # 垂直翻转
		data = TF.vflip(img=data)
		label = TF.vflip(img=label)
	if augCode == 3:  # 旋转45度
		data = TF.rotate(img=data, angle=45, resample=Image.NEAREST)
		label = TF.rotate(img=label, angle=45, resample=Image.NEAREST)
	if augCode == 4:  # 旋转-45度
		data = TF.rotate(img=data, angle=-45, resample=Image.NEAREST)
		label = TF.rotate(img=label, angle=-45, resample=Image.NEAREST)
	if augCode == 5:  # 亮度50%~150%随机值
		data = transforms.ColorJitter(brightness=0.5)(data)
	if augCode == 6:  # 对比度50%~150%随机值
		data = transforms.ColorJitter(contrast=0.5)(data)
	# data = transforms.ToTensor()(data)
	# label = torch.from_numpy(image2label(label,colormap))
	# label = transforms.ToTensor()(label)
	return data, label

def main():
    image_path = os.path.abspath('../cateimage/23/')
    label_path = os.path.abspath('../catelabel/23/')
    new_image_path = os.path.abspath('../cateimage/scale23/')
    new_label_path = os.path.abspath('../catelabel/scale23/')
    print("begin")

    files = os.listdir(image_path)
    for file in files:
        print(file)
        img, label = cv2.imread(os.path.join(image_path,file)), cv2.imread(os.path.join(label_path, file))
        data_tfs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = _mean, std = _std)])
        img = data_tfs(img)
        img = np.transpose(img.numpy(), (1, 2, 0))
        img, label = scalepadding(img, label)
        cv2.imwrite(os.path.join(new_image_path, file), img)
        cv2.imwrite(os.path.join(new_label_path, file), label)

    for file in files:
        name, expand = os.path.splitext(file)
        newname = name + '-aug'
        print(newname)
        augcode = random.randint(1, 6)
        img, label = Image.open(os.path.join(new_image_path,file)), Image.open(os.path.join(new_label_path, file))
        img, label = img_augment(img, label, augcode)  #
        print(type(img))
        print(type(label))
        # vutils.save_image(img, os.path.join(new_image_path, newname + expand), normalize = False)
        # vutils.save_image(label, os.path.join(new_label_path, newname + expand), normalize=False)
        img.save(os.path.join(new_image_path, newname + expand), quality=95)
        label.save(os.path.join(new_label_path, newname + expand), quality=95)

if __name__ == '__main__':
    main()
    print("end")