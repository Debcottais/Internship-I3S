import numpy as np

import skimage.filters
from skimage import io, filters 
import skimage.morphology
import skimage.exposure

import matplotlib.pyplot as plt
import seaborn as sns
import glob, os, sys, platform

from PIL import Image

rc={'lines.linewidth': 3, 'axes.labelsize': 18, 'axes.titlesize': 18}
sns.set(rc=rc)

filespath = os.path.join("/home/dcottais/Documents/Script_Scikit/data")

images = []
folder = filespath
for filename in os.listdir(folder):
    try:
        img = io.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    except:
        print('Cant import ' + filename)
images = np.asarray(images)

# plt.imshow(images[5])
# plt.show()

for img in images :
    new_image=[]
    im_cfp_filt = skimage.filters.gaussian(img, sigma=(5),multichannel=None)
    
## Thresholding 
# Compute Otsu thresholds for cfp
    # thresh_cfp_otsu = skimage.filters.threshold_otsu(im_cfp_filt)
# print(np.mean(thresh_cfp_otsu))
# resr=0.016085121746589937

    # thresh_cfp=0.016085121746589937
    # im_cfp_bw=im_cfp_filt>thresh_cfp
    # new_image.append(im_cfp_bw)
    new_image.append(im_cfp_filt)
print(new_image)
new_image=np.asarray(new_image)

# # Show the result

plt.imshow(images[2])
plt.show()
plt.imshow(new_image[2])
plt.show()

# http://scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html
# http://scikit-image.org/docs/dev/auto_examples/applications/plot_coins_segmentation.html#sphx-glr-auto-examples-applications-plot-coins-segmentation-py
