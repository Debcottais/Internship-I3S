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

tiff = tc.opentiff('treat01_33_R3D.dv - C=2.tif') #open img

for img in tiff:
    img_crop=img.cropped[1020,1024]

    cv2.imshow('cropped',img_crop)
    print(img_crop.shape) 
