from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import block_reduce
import glob, os, sys, platform
import matplotlib.pyplot as plt
from astropy.stats import SigmaClip
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.datasets import make_100gaussians_image
from photutils import Background2D, MedianBackground

image = 'treat01_33_R3D.dv - C=00017.tif'
image=Image.open(image)
width,height=image.size

image_crop=image.crop((4, 0, 1024, 1024)).save("crop.tif")
new_im="crop.tif"
img=io.imread(new_im)

data = img
ny, nx = data.shape
y, x = np.mgrid[:ny, :nx]
gradient =  x * y / 5000.
data2 = data + gradient
sigma_clip = SigmaClip(sigma=3.)
bkg_estimator = MedianBackground()
bkg = Background2D(data2, (170, 128), filter_size=(3, 5),
                   sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
norm = ImageNormalize(stretch=SqrtStretch())
plt.imshow(data2 - bkg.background, norm=norm, origin='lower',
           cmap='Greys_r')
plt.show()
