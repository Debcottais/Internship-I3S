import numpy as np 
import cv2
from skimage import io 
from skimage.measure import * 
from PIL import Image 
import matplotlib.pyplot as plt
import matplotlib as mpl 
import tiffcapture as tc
import imutils
import libtiff
from scipy.cluster import hierarchy
from scipy.ndimage import label 
from random import randint
import tifffile as tiff
import os, sys


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

# a = tiff.imread('treat01_33_R3D.dv - C=2.tif')
# print(a.shape)
# print(a.dtype)

############################################################333

tiff = tc.opentiff('treat01_33_R3D.dv - C=2.tif') #open img
# _, first_img = tiff.retrieve()

for img in tiff:
    cropped_image = img.crop((4, 0, 1024, 1024)) ## ne fonctionne pas :AttributeError: 'numpy.ndarray' object has no attribute 'crop'

    # img_crop=img[1024//1 -1024//1, 1024//1020: -1024//1020]
#     cv2.imshow('cropped',img_crop)
#     print(img_crop.shape) 

#     cv2.imshow('video',img)
#     cv2.waitKey(30)
# cv2.destroyAllWindows()



# for img in tiff:
#     # im=np.arange(img)
#     im=block_reduce(img,block_size=(170,128),func=np.sum,cval=0)
#     print(im)

# for img in tiff:
#     img_crop=img[1024: -1024//6,1024//8: -1024//8]
# #     # left=4
#     # top=1024
#     # width=1024-left
#     # height=1024
#     # box=(left,top,width,height)
#     # area=img.crop(box)
#     # print(np.array(img_crop))
#     # # cv2.imshow('cropped',img_crop)
#     # cv2.imshow('video',img)
#     # print(np.array(img))
    # im_crop=img.crop()
    # cv2.imshow('cropped',img_crop)
#     cv2.imshow('video',img)

    # tempimg = cv2.absdiff(first_img, img) # bkgnd sub
#     # plt.imshow(img) #ca va afficher toutes les images du tiff a chaque time point de la video
#     # plt.show()


    # _, tempimg = cv2.threshold(tempimg, 5, 255,
#     #     cv2.THRESH_BINARY) # convert to binary
    # cv2.imshow('background sub', tempimg)
#     # cv2.imshow('threshold',th3)
    # print(tempimg)



# tiff = tc.opentiff('treat01_33_R3D.dv - C=2.tif') #open img
# _, first_img = tiff.retrieve()
# def crop(image_path, coords, saved_location):
#     """
#     @param image_path: The path to the image to edit
#     @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
#     @param saved_location: Path to save the cropped image
#     """
#     image_obj = Image.open(image_path)
#     cropped_image = image_obj.crop(coords)
#     cropped_image.save(saved_location)
#     cropped_image.show()
 
 
# if __name__ == '__main__':
#     image = 'treat01_33_R3D.dv - C=2.tif'
#     crop(image, (1020, 1024, 1024, 1024), 'cropped.tif')

