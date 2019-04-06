import numpy as np

import skimage.filters
from skimage import io, filters 
import skimage.morphology
import skimage.exposure

import matplotlib.pyplot as plt
import seaborn as sns
rc={'lines.linewidth': 3, 'axes.labelsize': 18, 'axes.titlesize': 18}
sns.set(rc=rc)

# image='crop_image.tif'
# img=io.imread(image)
# print(img.shape)
# print(type(img))
# print(img)


# Load the phase contrast image.
im_phase = io.imread('crop_image.tif')

# def valid_imshow_data(data):
#     data = np.asarray(data)
#     if data.ndim == 2:
#         return True
#     elif data.ndim == 3:
#         if 3 <= data.shape[2] <= 4:
#             return True
#         else:
#             print('The "data" has 3 dimensions but the last dimension '
#                   'must have a length of 3 (RGB) or 4 (RGBA), not "{}".'
#                   ''.format(data.shape[2]))
#             return False
#     else:
#         print('To visualize an image the data must be 2 dimensional or '
#               '3 dimensional, not "{}".'
#               ''.format(data.ndim))
#         return False
# valid_imshow_data(im_phase)

# Display the image, set Seaborn style 'dark' to avoid grid lines
with sns.axes_style('dark',rc):
    # io.imshow(im_phase/im_phase.max())
    # io.show()
    # Get subplots
    fig, ax = plt.subplots(2, 2, figsize=(8,6))

# cells are 'white' on a black background 

    # Display various LUTs
    ax[0,0].imshow(im_phase, cmap=plt.cm.gray) #high pixel values are more white, and low pixel values are more black
    ax[0,1].imshow(im_phase, cmap=plt.cm.RdBu_r)
    ax[1,0].imshow(im_phase, cmap=plt.cm.viridis)
    ax[1,1].imshow(im_phase, cmap=plt.cm.copper)
    io.show()

# ####### segmentation ##########

#with CFP channel : channel0 like my image 
im_cfp = io.imread('crop_image.tif')

with sns.axes_style('dark'):
    plt.imshow(im_cfp, cmap=plt.cm.gray)
    plt.show()
    plt.imshow(im_cfp, cmap=plt.cm.viridis)
    # Add in a color bar
    plt.colorbar()
    plt.show()

# Perform the gaussian filter
im_cfp_filt = skimage.filters.gaussian(im_cfp, sigma=(5),multichannel=None)

# Show filtered image with the viridis LUT. 
with sns.axes_style('dark'):
    plt.imshow(im_cfp_filt, cmap=plt.cm.viridis)
    plt.colorbar()
    plt.show()

# ## Thresholding 

hist_cfp, bins_cfp=skimage.exposure.histogram(im_cfp_filt)
print(hist_cfp)
plt.fill_between(bins_cfp, hist_cfp,alpha=None)
plt.show()
plt.plot([0.0045, 0.0045], [0,50000], linestyle='-', marker='None', color='red')
plt.xlabel('pixel value')
plt.ylabel('count')
plt.show()


# large peak around 0.0038 ==> background
# the other peak around 0.006 ==> cells
# 0.0046 represents valley

# thresh_cfp=0.0046 #by eyes 

#with the print of compute Otsu thresholds for cfp we found a thresh value of :
thresh_cfp=0.00647242433585276
im_cfp_bw=im_cfp_filt>thresh_cfp

# Display phase and thresholded image
with sns.axes_style('dark'):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(im_cfp_filt, cmap=plt.cm.gray)
    ax[1].imshow(im_cfp_bw, cmap=plt.cm.gray)
    plt.show()

# Build RGB image by stacking grayscale images
im_cfp_rgb = np.dstack(3 * [im_cfp_filt / im_cfp_filt.max()])

# Saturate red channel wherever there are white pixels in thresh image
im_cfp_rgb[im_cfp_bw, 1] = 1.0

# Show the result
with sns.axes_style('dark'):
    plt.imshow(im_cfp_rgb)
    plt.show()


# Compute Otsu thresholds for cfp
thresh_cfp_otsu = skimage.filters.threshold_otsu(im_cfp_filt)
print(thresh_cfp_otsu)

### above thanks to http://justinbois.github.io/bootcamp/2016/lessons/l38_intro_to_image_processing.html 
#####################################################################################################
