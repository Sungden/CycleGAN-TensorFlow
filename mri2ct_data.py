import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
import imageio
import shutil
from shutil import copyfile
import cv2

MR='/data/pancreas_data/another_7_cases/cleaned/nii/'
CT='/data/ydeng1/cycle_gan/mri2ct/CT/'

MR_dst="/data/ydeng1/cycle_gan/MR_test_predict/"
CT_dst='/data/ydeng1/cycle_gan/mri2ct/testB/'

'''
image_dir='/data/pancreas_data/time_series_Liver_data/'
imglist=sorted(os.listdir(image_dir))
print(len(imglist))
for img in imglist:
  if 'Mr' in img:
    shutil.copyfile(os.path.join(image_dir,img), os.path.join(dst,img))
  if 'CT' in img:
    shutil.copyfile(os.path.join(image_dir,img), os.path.join(dst1,img))
'''

for i in sorted(os.listdir(MR))[2:7]:
  print(i,4444444444444444444444444444444)
  image=nib.load(os.path.join(MR,i)).get_data()
  print(image.shape)
  for j in range(image.shape[2]):
    if np.count_nonzero(image[:,:,j])>100 and np.max(image[:,:,j])>200:
      try:
        img=image[:,:,j].astype('float32')
        #img[img<=-100]=-100
        #img[img>=240]=240 
        name=i[0:4]+'_'+str(j)+'.png'
        img=cv2.resize(img,(256, 256))
        imageio.imsave(os.path.join(MR_dst,name),img)
      except:
        pass