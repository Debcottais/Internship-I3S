import cv2
import time 
import random
import imutils ## functions to make basic image processing (translation, rotation, skeletonization..)
from imutils.video import VideoStream 
from imutils.video import FPS
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os


cap=cv2.VideoCapture('/home/debby/Bureau/La cellule en mouvement.avi')

# read first frame
success, frame=cap.read()
#quit if unable to read the video file
if not success:
    print('Failed to read video')
    sys.exit(1)

while (cap.isOpened()):

    # capture frame-by-frame 
    ret, frame = cap.read()
    if ret == True:
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # display the resulting frame 
        cv2.imshow('frame',gray)
        plt.show(block=True)

        # Press Q on keyboard to  exit
        if cv2.waitKey(10)&0xFF==ord('q'):
            break
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows() 

# cv2.calcOpticalFlowPyrLK() ##function which allows to track feature points in a video 
# cv2.goodFeaturesToTrack() ## function which allows us to decide the points 


###############################################################################
## prog which created some frame : it works !

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name

# cap = cv2.VideoCapture('La cellule en mouvement.avi')

# try:
#     if not os.path.exists('data'):
#         os.makedirs('data')
# except OSError:
#     print ('Error: Creating directory of data')

# currentFrame=0
# while(True):
#     ret, frame=cap.read()

#     name='./data/frame'+str(currentFrame)+'.jpg'
#     print ('Creating ...'+name)
#     cv2.imwrite(name,frame)
    
#     currentFrame+=1
#     if not ret:
#         break

# cap.release()
# cv2.destroyAllWindows()


