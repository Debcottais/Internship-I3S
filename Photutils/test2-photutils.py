# pip3 install photutils


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import io 
from skimage.measure import block_reduce 
import glob, os, sys, platform
import cv2 
import scipy
from scipy import *
from skimage.util import view_as_blocks
from astropy.stats import SigmaClip
from photutils import Background2D, MedianBackground, ModeEstimatorBackground
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel
from astropy.modeling.models import Gaussian2D
from photutils import deblend_sources ##segmentation
from astropy.stats import gaussian_fwhm_to_sigma
from photutils import detect_sources
from photutils import detect_threshold
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize


# filepath = os.path.join("/home/dcottais/Documents/Internship-I3S/TestScriptOpenCV/treat01_33_R3D.dv\ -\ C=00017.tif")
image = 'treat01_33_R3D.dv - C=00017.tif'
image=Image.open(image)
width,height=image.size 
# print(width,height)

# plt.imshow(image)
# plt.title('Image 1024*1024')
# plt.show()

image_crop=image.crop((4, 0, 1024, 1024)).save("crop.tif")
new_im="crop.tif"
img=io.imread(new_im)
# img=Image.open(new_im)
# w,h=img.size 
# print(w,h)

# print(img.shape)
# plt.imshow(img)
# plt.title('Image cropped (1020*1024)')
# plt.show()

## Estimate local background :

# sigma_clip=SigmaClip(sigma=3.)
# bkg_estimator=MedianBackground()
# bkg=Background2D(img,(128, 170),filter_size=(5,3),sigma_clip=sigma_clip,bkg_estimator=bkg_estimator)
# print (bkg.background_median)
# plt.imshow(bkg.background)
# plt.show(block=True)


## Estimate local background :

ny, nx= img.shape
y, x= np.mgrid[:ny,:nx]
gradient=x*y / 5000
img2=img+gradient
plt.imshow(img2)
plt.title('Image+gradient')
plt.show()
sigma_clip=SigmaClip(sigma=3.)
bkg_estimator=MedianBackground()
# bkg_estimator=ModeEstimatorBackground()

## smoothing 
gauss_kernel = Gaussian2DKernel(3)
smoothed_data_gauss = convolve(img, gauss_kernel) 
plt.imshow(smoothed_data_gauss)
plt.title('Image smoothed')
plt.show()

bkg=Background2D(smoothed_data_gauss,(128, 170),filter_size=(5,3),filter_threshold=None,sigma_clip=sigma_clip,bkg_estimator=bkg_estimator)
print ('babckground median :',bkg.background_median)
print('Backgound local :', bkg.background_rms_median)
plt.imshow(bkg.background)
plt.title('Background')
plt.show()

## background substracted to the cropped image
Image=smoothed_data_gauss-bkg.background
plt.imshow(Image)
plt.title('Image with background substracted to the cropped image')
plt.show(block=True)
print('Background final :', Image)



## 'inversement' et watershed :
threshold = detect_threshold(Image, snr=3.)
sigma = 3.0 * gaussian_fwhm_to_sigma    # FWHM = 3.
kernel = Gaussian2DKernel(sigma, x_size=12, y_size=5)
kernel.normalize()
segm = detect_sources(Image, threshold, npixels=5, filter_kernel=kernel)
segm_deblend=deblend_sources(Image,segm,npixels=5,filter_kernel=kernel,nlevels=32,contrast=0.001)

plt.imshow(segm_deblend)
plt.title('Image segmentation')
plt.show()
# norm = ImageNormalize(stretch=SqrtStretch())
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
# ax1.imshow(img2)
# ax1.set_title('img2')
# ax2.imshow(segm_deblend)
# ax2.set_title('Segmentation Image')


# https://github.com/astropy/photutils
# +checker dans ma toolbar 
##############################################################################

