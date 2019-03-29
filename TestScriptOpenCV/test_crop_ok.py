from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob, os, sys, platform
import cv2 

#####################################################################
## This is working !!!
## Sauf que ca me les avait mises dans le dossier testscriptopencv 
## donc j'ai cree un dossier images_channel0_cropped pour les mettre dedans
## donc a voir si on peut mettre code pour que ca sauvegarde dans un dossier qu'on definit
## + voir pour faire un mix avec la def en dessous car plus propre joli !

filespath = os.path.join("/home/dcottais/Documents/Internship-I3S/TestScriptOpenCV/Images/images_channel0")

for file in os.listdir(filespath):
    if file.endswith(".tif"):
        filepath = os.path.join(filespath, file)
        im = Image.open(filepath)
        im.crop((4, 0, 1024, 1024)).save("crop"+file)


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
## tester ce code 
# from PIL import Image
# import os.path, sys

# path = "C:\\Users\\xie\\Desktop\\tiff\\Bmp"
# dirs = os.listdir(path)

# def crop():
#     for item in dirs:
#         fullpath = os.path.join(path,item)         #corrected
#         if os.path.isfile(fullpath):
#             im = Image.open(fullpath)
#             f, e = os.path.splitext(fullpath)
#             imCrop = im.crop((30, 10, 1024, 1004)) #corrected
#             imCrop.save(f + 'Cropped.bmp', "BMP", quality=100)

# crop()