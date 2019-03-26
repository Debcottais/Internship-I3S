import numpy as np 
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl 
import imutils
from scipy.cluster import hierarchy
from random import randint

cap=cv2.VideoCapture('La cellule en mouvement.avi')

######################################
### bounding box part 

# Specify the tracker type
trackerName='crst' ## Discriminative correlation filter tracker with channel and spatial reliability

# Create MultiTracker object 
trackers=cv2.MultiTracker_create()
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

#######################################
#### first step of the post doc script 

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
    erosion=cv2.erode(th3,kernel,iterations=1) #a voir si erosion surframe de depart ou sur frame apres filtre soit sur th3

    cv2.imshow('erosion',erosion)
## ou a la place de erosion : on fait un opening pour enlever les pixels qui pourrait etre en trop et donc confondus avec une cell
    kernel=np.ones((5,5),np.uint8) #uint8==>8 bit integer (represent values between 0 to 255)
    opening=cv2.morphologyEx(frame,cv2.MORPH_OPEN,kernel)

    # cv2.imshow('opening',opening)

## ou closing 
    kernel=np.ones((5,5),np.uint8) #uint8==>8 bit integer (represent values between 0 to 255)
    closing=cv2.morphologyEx(frame,cv2.MORPH_CLOSE,kernel)

    # cv2.imshow('closing',closing)
    cv2.imshow('cap',frame)

##############################################################
###### suite of bounding box part 

## draw bounding box for selection of ROI

    success, boxes=trackers.update(frame)

# loop over the bounding boxes and draw them on the frame
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(0) & 0xFF

    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        colors = []
    # select the bounding box of the object we want to track (make
    # sure you press ENTER or SPACE after selecting the ROI)
        box = cv2.selectROIs("Frame", frame, fromCenter=False, showCrosshair=True)
        box = tuple(map(tuple, box)) 
        for bb in box:
            tracker = OPENCV_OBJECT_TRACKERS[trackerName]()
            trackers.add(tracker, frame, bb)

# if you want to reset bounding box, select the 'r' key 
    elif key == ord("r"):
        trackers.clear()
        trackers = cv2.MultiTracker_create()

        box = cv2.selectROIs("Frame", frame, fromCenter=False, showCrosshair=True)
        box = tuple(map(tuple, box))
        for bb in box:
            tracker = OPENCV_OBJECT_TRACKERS[trackerName]()
            trackers.add(tracker, frame, bb)

    elif key == ord("q"):
        break
        # sys.exit(1)

    while cap.isOpened():
        success,frame=cap.read()
        if not success:
            break
        success,boxes=trackers.update(frame)
        
        for i, newbox in enumerate(boxes):
            p1=(int(newbox[0]), int(newbox[1]))
            p2=(int(newbox[0]+newbox[2]),int(newbox[1]+newbox[3]))
            cv2.rectangle(frame,p1,p2,colors[i],2,1)
        cv2.imshow('Multitracker',frame)


    ##################################################
    #### suite of the script post doc part 

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

## find contours in the thresholded image 
    # ret,contours,hierarchy=cv2.findContours(th3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # cnts=contours[0]
    # # cnts=imutils.grap_contours(cnts)
    # for cnts in contours:
    #     M=cv2.moments(cnts)
    #     if M["m00"] != 0:
    #         cX = int(M["m10"] / M["m00"])
    #         cY = int(M["m01"] / M["m00"])
    #     else:
    #         cX, cY = 0, 0
    #     cX=int(M["m10"]/M["m00"])
    #     cY=int(M["m01"]/M["m00"])
    # # draw the contour and center of the shape on the image
    # # cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
    #     cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
    #     cv2.putText(frame, "centroid", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    #     cv2.imshow("centroid",frame)
