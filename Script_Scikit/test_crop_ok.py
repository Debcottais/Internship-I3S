from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import io 
from skimage.measure import * 
import glob, os, sys, platform
import cv2 

#####################################################################
## This is working !!!
## Sauf que ca me les avait mises dans le dossier testscriptopencv 
## donc j'ai cree un dossier images_channel0_cropped pour les mettre dedans
## donc a voir si on peut mettre code pour que ca sauvegarde dans un dossier qu'on definit
## + voir pour faire un mix avec la def en dessous car plus propre joli !

# filespath = os.path.join("/home/dcottais/Documents/Internship-I3S/TestScriptOpenCV/Images")

ImageCropCFP=[]
ImageCropFRET=[]
ImageCropMOMP=[]
ImageCropPOL=[]       

def crop(filespath):
    for img in os.listdir(filespath):
        if img.endswith(".tif"):
            filepath = os.path.join(filespath, img)
            image_obj = Image.open(filepath)
            if filespath==os.path.join("/home/dcottais/Documents/Internship-I3S/TestScriptOpenCV/Images/Images_channel0"):
                image_crop=image_obj.crop((4, 0, 1024, 1024)).save("crop"+img)
                ImageCropCFP.append(image_crop)

        #     elif filespath==os.path.join("/home/dcottais/Documents/Internship-I3S/TestScriptOpenCV/Images/Images_channel1"):
        #         image_crop=image_obj.crop((4, 0, 1024, 1024)).save("crop"+img)
        #         ImageCropFRET.append(image_crop)
        #     elif filespath==os.path.join("/home/dcottais/Documents/Internship-I3S/TestScriptOpenCV/Images/Images_channel2"):
        #         image_crop=image_obj.crop((4, 0, 1024, 1024)).save("crop"+img)
        #         ImageCropMOMP.append(image_crop)
        #     elif filespath==os.path.join("/home/dcottais/Documents/Internship-I3S/TestScriptOpenCV/Images/Images_channel3"):
        #         image_crop=image_obj.crop((4, 0, 1024, 1024)).save("crop"+img)
        #         ImageCropPOL.append(image_crop)           

crop("/home/dcottais/Documents/Internship-I3S/TestScriptOpenCV/Images/Images_channel0")
# print(ImageCropCFP)

crop("/home/dcottais/Documents/Internship-I3S/TestScriptOpenCV/Images/Images_channel1")
crop("/home/dcottais/Documents/Internship-I3S/TestScriptOpenCV/Images/Images_channel2")
crop("/home/dcottais/Documents/Internship-I3S/TestScriptOpenCV/Images/Images_channel3")

# filespath1 = os.path.join("/home/dcottais/Documents/Internship-I3S/TestScriptOpenCV/images_channel1_cropped")


# def blockReduce():
#     for imgCrop in ImageCropCFP:
#         # filepath1=os.path.join(filespath1,imgCrop)
#         image=Image.open(imgCrop)
#         new_image=np.array(image)
#         new_image=block_reduce(new_image,block_size=(6,8),func=np.sum,cval=0)
#         im=signal.convolve2d(new_image,np.array(new_image),boundary=fill,mode='same') 

# blockReduce()

# def convolution():
#     s=new_image.shape
#     py=(H.shape[0]-1)    


##################################################################
### Work for one image ##### 

# Donc a voir pour faire comme au-dessus mais dans cette def car plus joli 

# def crop(image_path, coords, saved_location):
#     """
#     @param image_path: The path to the image to edit
#     @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
#     @param saved_location: Path to save the cropped image
#     """
#     image_obj = Image.open(image_path)
#     image_obj.show()
#     cropped_image = image_obj.crop(coords)
#     cropped_image.save(saved_location)
#     cropped_image.show()

 
 
# if __name__ == '__main__':
#     image = 'treat01_33_R3D.dv - C=00017.tif'
#     crop(image, (4, 0, 1024, 1024), 'cropped.tif')
#     print(image.shape)
#     print(image.dtype)

#####################################################################
## Coupage de l'image en 64 block de 170*128 ##

