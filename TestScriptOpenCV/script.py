import numpy as np 
import cv2
from skimage import io 
from skimage.measure import * 
from PIL import Image as pilimage 
import matplotlib.pyplot as plt
import matplotlib as mpl 
import tiffcapture as tc
import imutils
import libtiff
from scipy.cluster import hierarchy
from scipy.ndimage import label 
from random import randint
import sys
import tifffile as tiff


###############################################################################
# using scikit image  which opens multiple slices from a tiff stack by default. 
# Here, we can load a tiff stack into a Numpy array :

# tiff = io.imread('treat01_33_R3D.dv - C=2.tif')
# print(tiff.shape)

# # if we want to get a single slice from the stack :
# print (tiff[1].shape)


# tiff = tc.opentiff('treat01_33_R3D.dv - C=2.tif')
# plt.imshow(tiff.read()[1])
# plt.show()
# tiff.release()

a = tiff.imread('treat01_33_R3D.dv - C=2.tif')
print(a.shape)
print(a.dtype)

tiff = tc.opentiff('treat01_33_R3D.dv - C=2.tif') #open img
_, first_img = tiff.retrieve()

# for img in tiff:
#     # im=np.arange(img)
#     im=block_reduce(img,block_size=(170,128),func=np.sum,cval=0)
#     print(im)

for img in tiff:
    # img_crop=img[1024: -1024//6,1024//8: -1024//8]
    left=4
    top=1024
    width=1024-left
    height=1024
    box=(left,top,width,height)
    area=img.crop(box)
    cv2.imshow('cropped',area)
    cv2.imshow('video',img)
    # tempimg = cv2.absdiff(first_img, img) # bkgnd sub
#     # plt.imshow(img) #ca va afficher toutes les images du tiff a chaque time point de la video
#     # plt.show()


    # _, tempimg = cv2.threshold(tempimg, 5, 255,
#     #     cv2.THRESH_BINARY) # convert to binary
    # cv2.imshow('background sub', tempimg)
#     # cv2.imshow('threshold',th3)
    # print(tempimg)
    cv2.waitKey(30)
cv2.destroyWindow('video')




