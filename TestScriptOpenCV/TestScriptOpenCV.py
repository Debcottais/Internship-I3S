import numpy as np 
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl 


FILE = "cellule.jpg"

#read the image and filter (image processing)
img=cv2.imread(FILE,0)
print(img)

# #gaussian filter
blur=cv2.GaussianBlur(img,(5,5),0)
ret3,th3=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

##plot the filtering process
plt.figure(figsize=(15,5))
images = [blur, 0, th3]
titles = ['Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
plt.subplot(1,3,1),plt.imshow(images[0],'gray')
# plt.show(block=True) ## to display the plot 
plt.title(titles[0]), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.hist(images[0].ravel(),256)
# plt.show(block=True)
plt.title(titles[1]), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(images[2],'gray')
plt.title(titles[2]), plt.xticks([]), plt.yticks([])
plt.show(block=True) ## to dislay on the same window all the plots 

## count the number of contours in the filtered image
im = cv2.imread(FILE)
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
img = cv2.drawContours(img, contours, -1, (255,255,255), 3)
print(plt.figure(figsize=(10,10)))
plt.subplot(1,2,1),plt.title('Original Image'),plt.imshow(im)#,'red')
plt.subplot(1,2,2),plt.title('OpenCV.findContours'),plt.imshow(img,'gray')#,'red')
plt.show(block=True)

print('number of detected contours:',len(contours))


