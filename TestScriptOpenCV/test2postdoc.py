import numpy as np 
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl 
import imutils
from scipy.cluster import hierarchy
from scipy.ndimage import label 
from random import randint
import sys

cap=cv2.VideoCapture('La cellule en mouvement.avi')

while(True):
    ret,frame=cap.read()
    frame = imutils.resize(frame, width=600)

##### 1-Estimation local background on channel CFP (the first)
##### 2-Lissage channel1 (step3)

## Gaussian filter+otsu's threshold 
    img_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(img_gray,(0,0),5)
    ret3,th2=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #avec le otsu, permet une precision sur cell 

    th3=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    cv2.imshow('Gaussian filter+otsu''s threshold',th2)
    cv2.imshow('Gaussian filter+adaptative threshold',th3)
    
##### 3-Inversion et watershed channel1 (step 4 and 5)
    msk=cv2.watershed(-th3)


    cv2.imshow('cap',frame)

    k=cv2.waitKey(30) & 0xff #plus on augmente la key plus la video va doucement
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()