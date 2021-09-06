# 1.dcm2nii.py 

实现将dcm图像转换成nii图像

可以根据代码中写的，也可以直接根据dicom2nifti包，比如：

```python
import dicom2nifti

original_dicom_directory = r".\Dicomdataset"
output_file = r'./result/test.nii'  #.nii.gz也可以
dicom2nifti.dicom_series_to_nifti(original_dicom_directory, output_file, reorient_nifti=True)
```

# 2.nii2img.py

此函数实现将nii.gz数据转换成3个维度的图片信息

# 3.scalepadding.py

本函数实现图片的padding、扩充预处理

# 4 . category.py

统计图像的锥体，属于颈椎、颈胸椎、颈胸腰椎、胸腰椎
思想：根据标签的数值来判断属于哪一个部位的