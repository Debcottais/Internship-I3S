import cv2
import time 
import imutils ## functions to make basic image processing (translation, rotation, skeletonization..)
from imutils.video import VideoStream 
from imutils.video import FPS
from random import randint 
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


# Specify the tracker type
trackerType='CSRT' ## Discriminative correlation filter tracker with channel and spatial reliability

# Create MultiTracker object 
multiTrackers=cv2.MultiTracker_create()

## Selection of objects 
bboxes=[]
colors=[]

while True:
    frame=imutils.resize(frame, width=600)
    success, bboxes=multiTrackers.update(frame)
    ## selectROI allows to open a graphical interface (GUI) to select bounding boxes (particles of interest)
    bbox=cv2.selectROI('MultiTracker',frame) 
    bboxes.append(bbox)
    colors.append((randint(0,255),randint(0,255),randint(0,255)))
    print ("Press q to quit selecting boxes and start tracking")
    print ('Press any other key to select next object')
    key=cv2.waitKey(0) & 0xFF
    if (key==113): 
        break 
print ('Selected bounding boxes {}'.format(bboxes))


# Initialize MultiTracker

for bbox in bboxes:
    multiTracker=OPENCV_OBJECT_TRACKERS[trackerType]()
    multiTrackers.add(multiTracker, frame, bbox)

cap.release()
cv2.destroyAllWindows()
    

# https://www.learnopencv.com/multitracker-multiple-object-tracking-using-opencv-c-python/