import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
import imageio
import shutil
from shutil import copyfile
import cv2


png_dir="/data/ydeng1/cycle_gan/predict_nii_data/c_p7/"
nii_dir='/data/ydeng1/cycle_gan/predict_nii_data/'

png_list=os.listdir(png_dir)
#print(png_list)
png_list.sort()
png_list.sort(key=lambda x: int(x[5:-4]))
print(png_list)
data=np.zeros((384,384,len(png_list)))
for img,i in zip(png_list,range(data.shape[2])):
  print(img)
  image=cv2.imread(os.path.join(png_dir,img))
  image=image[:,:,0]
  image_1=cv2.resize(image,(384, 384))
  data[:,:,i]=image_1
  
MR = nib.Nifti1Image(data, np.eye(4))
nib.save(MR, os.path.join(nii_dir, 'c_p7.nii'))