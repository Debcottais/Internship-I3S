from PIL import Image

import numpy as np 

import matplotlib.pyplot as plt

from skimage import io, filter 
from skimage.measure import * 
from skimage.util import *
from skimage.util.shape import view_as_blocks
from skimage.morphology import *
from skimage.filters import gaussian_filter, median 

from scipy import ndimage

import os, sys


# using scikit image  which opens multiple slices from a tiff stack by default. 
# Here, we can load a tiff stack into a Numpy array :

# tiff = io.imread('treat01_33_R3D.dv - C=2.tif')
# print(tiff.shape)

###############################################
## test of script for only one image to begin 

# ## load image
# image='treat01_33_R3D.dv - C=00017.tif'
# image=io.imread(image)
# print(image)
# print(np.max(image))
# plt.imshow(image)
# plt.show()
# ## crop image to have 1020*1024
# cropped=image[0:1024,4:1024]
# print(cropped.shape)
# img=io.imsave('crop_image.tif',cropped)
# new_img='crop_image.tif'
# ## load the new image 
# image=io.imread(new_img)


image='crop_image.tif'
img=io.imread(image)
print(img.shape)
print(type(img))
print(img)

def block_reduce(image, block_size, func=np.mean, cval=0):
    """Down-sample image by applying function to local blocks.
    Parameters
    ----------
    image : ndarray
        N-dimensional input image.
    block_size : array_like
        Array containing down-sampling integer factor along each axis.
    func : callable
        Function object which is used to calculate the return value for each
        local block. This function must implement an ``axis`` parameter such
        as ``numpy.sum`` or ``numpy.min``.
    cval : float
        Constant padding value if image is not perfectly divisible by the
        block size.
    Returns
    -------
    image : ndarray
        Down-sampled image with same number of dimensions as input image.
    """
    # if len(block_size) != image.ndim:
    #     raise ValueError("`block_size` must have the same length "
    #                      "as `image.shape`.")

    # pad_width = []
    # for i in range(len(block_size)):
    #     if block_size[i] < 1:
    #         raise ValueError("Down-sampling factors must be >= 1. Use "
    #                          "`skimage.transform.resize` to up-sample an "
    #                          "image.")
    #     if image.shape[i] % block_size[i] != 0:
    #         after_width = block_size[i] - (image.shape[i] % block_size[i])
    #     else:
    #         after_width = 0
    #     pad_width.append((0, after_width))

    # image = np.pad(image, pad_width=pad_width, mode='constant',
    #                constant_values=cval)

    blocked = view_as_blocks(image, block_size)

    return func(blocked, axis=tuple(range(image.ndim, blocked.ndim)))

var=block_reduce(image=img,block_size=(128,170))

print(var.shape)
# print(var[7,5])
plt.imshow(var)
plt.show()

def background(var):
    for block in var :
        img_filt=gaussian_filter(block,sigma=(1,3,5),multichannel=None)
    filtered=view_as_blocks(img_filt)
    return func(filtered, axis=tuple(range(image.ndim, blocked.ndim)))

background(var)
