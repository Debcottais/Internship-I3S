from skimage import io 
import matplotlib.pyplot as plt
import numpy as np
from math import exp 
from skimage.measure import block_reduce, label
import skimage as im_
import scipy.interpolate as in_
from skimage import data
from scipy.signal.signaltools import convolve2d
from scipy.signal.windows import gaussian
from skimage.morphology import watershed, skeletonize
from skimage.filters import rank
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from matplotlib.pyplot import hist
from scipy.ndimage.measurements import histogram
from numpy.lib.function_base import percentile


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


im = io.imread('crop_image.tif') #c'est la 17 du canal 0
im2 = io.imread('treat01_33_R3D.dv - C=10018.tif')
im2=im2[0:1024,4:1024]

# print(im)
# print(im.shape)
# im=im_.color.rgb2gray(data.astronaut())*1000

def machin(a, axis=None, dtype=None, out=None, keepdims=False):
    print(a.shape,axis)
    return np.mean(a,axis=axis, dtype=dtype,out=out,keepdims=keepdims)



def estim_bg(image,blocksize):
    bg=np.empty((image.shape[0]//blocksize[0],image.shape[1]//blocksize[1]))

    for rows in range(bg.shape[0]):
        for cols in range(bg.shape[1]):
            block=image[rows*blocksize[0]:(rows+1)*blocksize[0],cols*blocksize[1]:(cols+1)*blocksize[1]]
            values_histo, bin_edges=np.histogram(np.log(block.flatten()+1)) #bins=block.size//100 
            values_histo=np.convolve(values_histo,[1/3,1/3,1/3],mode='same')
            max_value=np.argmax(values_histo)
            bg[rows,cols]=exp(bin_edges[max_value])
    return bg
def RescaledImage(image, new_size):
    #
    rows = tuple(range(image.shape[0]))
    cols = tuple(range(image.shape[1]))

    new_rows, new_cols = np.meshgrid(np.linspace(0, rows[-1], new_size[0]), #rows[-1]-1 #version d'avant
                                     np.linspace(0, cols[-1], new_size[1]), #cols[-1]-1 #vers d'avant
                                      indexing = 'ij')

    new_grid = np.dstack((new_rows.flat, new_cols.flat))

    return np.reshape(in_.interpn((rows, cols), image, new_grid, method = 'splinef2d'), new_size)
backgd=RescaledImage(estim_bg(im,(128,170)),im.shape)   
backgd2=RescaledImage(estim_bg(im2,(128,170)),im.shape)  

fig, ax = plt.subplots(1, 4, figsize=(20, 4))
ax[0].imshow(im)
ax[1].imshow(estim_bg(im,(128,170)))
ax[2].imshow(backgd)
ax[3].imshow(im-backgd)
# print(im-backgd)
plt.show()


### prepare watershed for further segmentation

def fspecial_gauss(size, sigma):
    
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()
gauss=fspecial_gauss(12,5)

# https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python

image=convolve2d((im-backgd),gauss,'same')
image2=convolve2d((im2-backgd),gauss,'same')

fig, ax = plt.subplots(1, 2, figsize=(20, 4))
ax[0].imshow(im-backgd)
ax[1].imshow(image)
plt.show()

#### test watershed with mahotas module python
# import mahotas 

# threshed = image > image.mean()
# distance = mahotas.stretch(mahotas.distance(threshed))
# bc = np.ones((3,3))
# maxima = mahotas.morph.regmax(distance, Bc=bc)
# spots, n_spots = mahotas.label(maxima, Bc=bc)
# surface = (distance.max() - distance)
# areas = mahotas.cwatershed(surface, spots)
# areas *= threshed

# fig, ax = plt.subplots(1, 2, figsize=(20, 4))
# ax[0].imshow(im)
# ax[1].imshow(areas)
# plt.show()

###########################################
### now binarization of my image 
from skimage.filters import threshold_otsu, threshold_local
global_thresh = threshold_otsu(image)
binary_global = image > global_thresh

block_size = 171 #63
binary_local = threshold_local(image, block_size, offset=10)
print('binary_local : ',binary_local)
binary_adaptive2 = threshold_local(image2, block_size, offset=10)


fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
ax0, ax1, ax2 = axes
plt.gray()

ax0.imshow(image)
ax0.set_title('Image')

ax1.imshow(binary_global)
ax1.set_title('Global thresholding')

ax2.imshow(binary_local)
ax2.set_title('Local thresholding')

for ax in axes:
    ax.axis('off')

plt.show()

### binarisation 2 avec code matlab 

# threshold=3*percentile(abs(image[:]),20)
# print ('threshold : ',threshold)

# msk=image>threshold

# fig, axes = plt.subplots(nrows=2, figsize=(7, 8))
# ax0, ax1 = axes
# plt.gray()

# ax0.imshow(image)
# ax0.set_title('Image')

# ax1.imshow(msk)
# ax1.set_title('thresholding')

# plt.show()

## marche beaucoup moins bien que la methode au-dessus 
# donnne 41 elements contre 52 pour l'autre


################################
local_maxi = peak_local_max(-image, indices=False)
# print(local_maxi)
markers = ndi.label(local_maxi)[0]
labels = watershed(-image, markers, mask=binary_global) #or mask=msk
print(labels)
# labels2 = watershed(-image2, markers, mask=binary_adaptive2) #or mask=msk


fig, ax = plt.subplots(1, 2, figsize=(20, 4))
ax[0].imshow(image)
ax[1].imshow(labels)
plt.show()

labeled_array,num_features=ndi.measurements.label(labels) #bwlabel
print(labeled_array)
print('The number of objects which were found is : ', num_features) 

### threshold based on pixel-to-pixel variability 
t=3*np.percentile(np.abs(image[:]),20)
print(t)
t2=3*np.percentile(np.abs(image2[:]),20)

msk=labels>t
print(msk)
# msk2=labels2>t2

plt.imshow(msk)
plt.show()
msk[labels==0]=0
# msk2[labels2==0]=0

# Return the skeleton of a binary image.
msk=skeletonize(msk) # bwmorph=skeletonize
plt.imshow(msk)
plt.show()
# msk2=skeletonize(msk2) # bwmorph=skeletonize

labeled_array,num_features=ndi.measurements.label(msk) #bwlabel
print(labeled_array)
print('The number of objects which were found is : ', num_features) 

###### align channels to correct for chromatic aberration

def getshift(image_ref, image, max_shift):
    steps=[floor(max_shift/4), 1, 1, 1]
    lpf = lowpass