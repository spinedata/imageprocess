'''
    此函数实现对图像锥体进行统计，因为26个类别类别太多，且样本不平衡，因此需要拆分出来，针对不同的类别方位而进行分类
    统计图片中有多少腰椎图像、有多少颈椎图像、有多少胸椎图像  颈椎：1-7  胸椎：8-19 腰椎：20-25（没有标注骶骨）

    算法流程：
    统计这个图片的像素值，将像素值按照从小到大排序；
    如果第二个像素值颈椎：
        最后一个像素值属于颈椎，则将它移动到颈椎文件，并将此名字的图像文件移动到对应的文件中
        如果最后一个像素值属于胸椎，则将它移动到颈胸椎文件，并将此名字的图像文件移动到对应的文件中
    如果第二个像素值属于胸椎：
        如果最后一个像素值属于胸椎：移动到胸椎文件，并将此名字的图像文件移动到对应的文件中
        如果最后一个像素值属于腰椎，移动到胸腰椎文件，并将此名字的图像文件移动到对应的文件中
    如果第二个像素值属于腰椎：
        则将它移动到腰椎文件，并将此名字的图像文件移动到对应的文件中
'''
import os    #对大量文件，大量路径进行操作
import glob
import shutil  #高级的 文件、文件夹、压缩包 处理模块
from typing import Union
from PIL import Image
import numpy as np


def catemove(label, image_path, label_path, cateimage, catelabel, dirname):
    # 复制名为label的标签到catelabel
    shutil.copy(os.path.join(label_path, label), os.path.join(catelabel, dirname))
    # 找到对应的名为label的image复制到cateimage
    shutil.copy(os.path.join(image_path, label), os.path.join(cateimage, dirname))
def catefunc(image_path, label_path, cateimage, catelabel):
    labels = os.listdir(label_path)
    for label in labels:
        fname, _ = os.path.splitext(label)
        img = Image.open(os.path.join(label_path, label))
        arr = np.array(img)  # 将img的格式转换成数组，以便后面矩阵展平
        pixel = set()
        for j in arr.flatten():
            pixel.add(j)  #不会重复
        print(sorted(pixel))
        pixellist = list(sorted(pixel))


        if pixellist[1] >= 1 and pixellist[1] <= 7:
            if pixellist[-1] >= 1 and pixellist[-1] <= 7:
                dirname = '1'  #颈椎
            elif pixellist[-1] >= 8 and pixellist[-1] <= 19:
                dirname = '12'  #颈胸椎
            elif pixellist[-1] >= 20 and pixellist[-1] <= 25:
                dirname = '123' #颈胸腰椎
        else:
            dirname = '23'  #胸腰椎
        catemove(label, image_path, label_path, cateimage, catelabel, dirname)
    print("end")

def main():
    image_path = os.path.abspath('../image/')
    label_path = os.path.abspath('../label/')
    cateimage = os.path.abspath('../cateimage/')
    catelabel = os.path.abspath('../catelabel/')
    catefunc(image_path, label_path, cateimage, catelabel)

if __name__ == '__main__':
    main()