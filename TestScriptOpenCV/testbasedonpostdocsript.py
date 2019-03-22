import numpy as np 
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl 
import imutils

cap=cv2.VideoCapture('La cellule en mouvement.avi')

## background substraction 
fgbg=cv2.bgsegm.createBackgroundSubtractorMOG()

while(True):
    ret,frame=cap.read()
    frame = imutils.resize(frame, width=600)
    fgmask=fgbg.apply(frame)

    # cv2.imshow('frame',fgmask)

## Gaussian filter+otsu's threshold 
    blur=cv2.GaussianBlur(fgmask,(5,5),0)
    ret3,th3=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #avec le otsu, permet une precision sur cell 

    # cv2.imshow('gaussian+otsu''s threshold',th3)

## erosion
    kernel=np.ones((5,5),np.uint8) #uint8==>8 bit integer (represent values between 0 to 255)
    erosion=cv2.erode(frame,kernel,iterations=1) #a voir si erosion surframe de depart ou sur frame apres filtre soit sur th3

    # cv2.imshow('erosion',erosion)
## ou a la place de erosion : on fait un opening pour enlever les pixels qui pourrait etre en trop et donc confondus avec une cell
    kernel=np.ones((5,5),np.uint8) #uint8==>8 bit integer (represent values between 0 to 255)
    opening=cv2.morphologyEx(frame,cv2.MORPH_OPEN,kernel)

    cv2.imshow('opening',opening)

## ou closing 
    kernel=np.ones((5,5),np.uint8) #uint8==>8 bit integer (represent values between 0 to 255)
    closing=cv2.morphologyEx(frame,cv2.MORPH_CLOSE,kernel)

    cv2.imshow('closing',closing)
    cv2.imshow('cap',frame)

    k=cv2.waitKey(50) & 0xff #plus on augmente la key plus la video va doucement
    if k==47:
        break

cap.release()
cv2.destroyAllWindows()

    ################################################
########### avant ca tout marche ##################
#####################################################




    # opening = cv2.morphologyEx(filtered.copy(), cv2.MORPH_OPEN, kernel, iterations = 2)
    # sure_bg=cv2.erode(opening,kernel,iterations=3)

    # dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    # ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # sure_fg = np.uint8(sure_fg)
    # unknown = cv2.subtract(sure_bg,sure_fg)

    
    # ret,markers=cv2.connectedComponents(fgmask)
    # markers=markers+1
    # markers[fgmask==255]=0

    # markers=cv2.watershed(frame,markers)
    # frame[markers==-1]=[255,0,0]


    # cv2.imshow('watershed',markers)


    # cv2.imshow('threshold',filtered)


