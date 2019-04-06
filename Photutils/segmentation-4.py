from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import io 
from skimage.measure import block_reduce 
import glob, os, sys, platform
import matplotlib.pyplot as plt
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.datasets import make_100gaussians_image
from photutils import Background2D, MedianBackground
from photutils import detect_threshold, detect_sources, deblend_sources
from photutils import source_properties
from photutils import EllipticalAperture

image = 'treat01_33_R3D.dv - C=00017.tif'
image=Image.open(image)
width,height=image.size 

image_crop=image.crop((4, 0, 1024, 1024)).save("crop.tif")
new_im="crop.tif"
img=io.imread(new_im)

data = img

bkg_estimator = MedianBackground()
bkg = Background2D(data, (170, 128), filter_size=(5, 3),
                   bkg_estimator=bkg_estimator)
threshold = bkg.background + (2. * bkg.background_rms)
sigma = 5.0 * gaussian_fwhm_to_sigma    # FWHM = 3.
kernel = Gaussian2DKernel(sigma, x_size=5, y_size=12)
kernel.normalize()
npixels = 5
segm = detect_sources(data, threshold, npixels=npixels,
                      filter_kernel=kernel)
segm_deblend = deblend_sources(data, segm, npixels=npixels,
#                                filter_kernel=kernel, nlevels=32,
#                                contrast=0.001)
cat = source_properties(data, deblend_sources)
r= 3.    # approximate isophotal extent
apertures = []
for obj in cat:
    position = (obj.xcentroid.value, obj.ycentroid.value)
    a = obj.semimajor_axis_sigma.value * r
    b = obj.semiminor_axis_sigma.value * r
    theta = obj.orientation.value
    apertures.append(EllipticalAperture(position, a, b, theta=theta))
norm = ImageNormalize(stretch=SqrtStretch())
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
ax1.imshow(data, origin='lower', cmap='Greys_r', norm=norm)
ax1.set_title('Data')
ax2.imshow(deblend_sources, origin='lower',
           cmap=deblend_sources.cmap(random_state=12345))
ax2.set_title('Segmentation Image')
for aperture in apertures:
    aperture.plot(color='white', lw=1.5, ax=ax1)
    aperture.plot(color='white', lw=1.5, ax=ax2)
plt.tight_layout()
plt.show()