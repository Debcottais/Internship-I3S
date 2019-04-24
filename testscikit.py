import numpy as np
import matplotlib.pyplot as plt
from math import exp 

from scipy import ndimage as ndi
from scipy.signal.signaltools import convolve2d
import scipy.interpolate as in_

from skimage.morphology import watershed, skeletonize, reconstruction
from skimage.feature import peak_local_max
from skimage import data, measure, color, filters
import skimage as im_
from skimage import io 

# image=im_.color.rgb2gray(data.astronaut())
im = io.imread('crop_image.tif') #c'est la 17 du canal 0
# image = io.imread('crop_image-binaire_inverse.tif') #16bits mais 3 canaux
# image = color.rgb2gray(image)
# image = io.imread('crop_image_binaire.tif') #it's work !!
# image=io.imread('crop_image_binaire2.tif')
print(im.shape)
print(im.dtype) #8 bits now because binarisation 

#### bg mesh ##############################################

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

image_bg = im-backgd

image_bg = filters.gaussian(image_bg, sigma=9)


### prepare watershed for further segmentation #############

# def fspecial_gauss(size, sigma):
    
#     """Function to mimic the 'fspecial' gaussian MATLAB function
#     """

#     x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
#     g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
#     return g/g.sum()
  
# gauss=fspecial_gauss(12,5)

# https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python


####### Now we want to separate the two objects in image
##### Generate the markers as local maxima of the distance 
##################### to the background

# image_flt=convolve2d(image_bg,gauss,'same')
# plt.imshow(image_flt)
# plt.show()

# distance = ndi.distance_transform_edt(image)
# print(distance)

local_maxi = peak_local_max(-image_bg, indices=False, footprint=np.ones((3, 3)),
                            labels=None) #labels=image pr binaire 
# local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
plt.imshow(local_maxi)
plt.show()
markers = ndi.label(local_maxi)[0]
labels = watershed(-image_bg, markers, mask=image_bg, watershed_line = True)
# labels = watershed(-distance, markers, mask=image)

plt.imshow(labels)
plt.show()
# print(labels)

fig, axes = plt.subplots(ncols=3, figsize=(20, 4), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image_bg, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title('Overlapping objects')
# ax[1].imshow(image_flt, cmap=plt.cm.gray, interpolation='nearest')
# ax[1].imshow(-distance, cmap=plt.cm.gray, interpolation='nearest')
# ax[1].set_title('Distances')
ax[1].imshow(labels, cmap=plt.cm.nipy_spectral, interpolation='nearest')
ax[1].set_title('Separated objects')
ax[2].imshow(im)
ax[2].set_title('original')
for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()

######## threshold based on pixel-to-pixel variability

t=3*np.percentile(np.abs(image_bg[:]),20,interpolation='nearest')
print(t)
msk=image_bg>t
print(msk)
plt.imshow(msk)
plt.show()
msk[labels==0]=0

### Return the skeleton of a binary image.
msk=skeletonize(msk) # bwmorph=skeletonize
plt.imshow(msk)
plt.show()

labeled_array,num_features=ndi.measurements.label(msk) #bwlabel
print(labeled_array.shape)
print('The number of objects which were found is : ', num_features) 


####### align channels to correct for chromatic aberration
# dx, dy = getshift(image_bg, image_bgB, 5)

####### proprieties ##############

properties = []
results = {'fr':[],'b':[], 'x':[], 'y':[], 'area':[],
            'prev':[], 'next':[], 'momp':[], 'edge':[], 'rfp':[]}
rp = measure.regionprops(labeled_array)
print(rp[6]['centroid'])
print(rp[6]['area'])
print(rp[6].coords)
print(rp[6].label)

# res
# (3.0, 385.0)
# 1
# [[  3 385]]
# 7