#!/usr/bin/env python

# import the necessary packages
from skimage.measure import structural_similarity as ssim
from skimage import color
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import sys
import sys
import math
import Image

eyecascPath = sys.argv[1]
eyeCascade = cv2.CascadeClassifier(eyecascPath)
#Recorded positions for left eye (their right) using Chien Ming's Face
aX = 38
aY = 103
bX = 92
bY = 60

#right eye (their left)
cX = 110
cY = 103
dX = 160
dY = 60


def isInEyeBox(eye, x, y):


    if eye == "left":
        return (aX <= x <= bX and bY <= y <= aY)
    if eye == "right":
        return (cX <= x <= dX and dY <= y <= cY)
    return false

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def compare_images(imageA, imageB):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(color.rgb2gray(imageA), color.rgb2gray(imageB))
    if(m>500):print(m)
    if(s<0.85): print(s)

    ###3
def Distance(p1,p2):
      dx = p2[0] - p1[0]
      dy = p2[1] - p1[1]
      return math.sqrt(dx*dx+dy*dy)

def ScaleRotateTranslate(image, angle, center = None, new_center = None, scale = None, resample=Image.BICUBIC):
      if (scale is None) and (center is None):
        return image.rotate(angle=angle, resample=resample)
      nx,ny = x,y = center
      sx=sy=1.0
      if new_center:
        (nx,ny) = new_center
      if scale:
        (sx,sy) = (scale, scale)
      cosine = math.cos(angle)
      sine = math.sin(angle)
      a = cosine/sx
      b = sine/sx
      c = x-nx*a-ny*b
      d = -sine/sy
      e = cosine/sy
      f = y-nx*d-ny*e
      return image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=resample)

def CropFace(image, eye_left=(0,0), eye_right=(0,0), offset_pct=(0.2,0.2), dest_sz = (70,70)):
      # calculate offsets in original image
      offset_h = math.floor(float(offset_pct[0])*dest_sz[0])
      offset_v = math.floor(float(offset_pct[1])*dest_sz[1])
      # get the direction
      eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
      # calc rotation angle in radians
      rotation = -math.atan2(float(eye_direction[1]),float(eye_direction[0]))
      # distance between them
      dist = Distance(eye_left, eye_right)
      # calculate the reference eye-width
      reference = dest_sz[0] - 2.0*offset_h
      # scale factor
      scale = float(dist)/float(reference)
      # rotate original around the left eye
      image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
      # crop the rotated image
      crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v)
      crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)
      image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0]+crop_size[0]), int(crop_xy[1]+crop_size[1])))
      # resize it
      image = image.resize(dest_sz, Image.ANTIALIAS)
      return image
def updateModel(file):
    #####
    # load the images -- the original, the original + contrast,
    # and the original + photoshop
    filename = file
    folder = "faces/"
    end = ".jpg"
    original = cv2.imread(folder + filename + end)
    # convert the images to grayscale
    eyes_pos = {
        'left_x' : None,
        'left_y' : None,
        'right_x' : None,
        'right_y' : None
    }
    eyes = eyeCascade.detectMultiScale(
        original,
        scaleFactor=1.01,
        minNeighbors=10,
        minSize=(10,10),
        maxSize=(40,40)
        )
    for (ox, oy, w, h) in eyes:
        x = ox + (w/2)
        y = oy + (h/2)
        if(isInEyeBox("left", x, y)):
            eyes_pos['left_x'] = x
            eyes_pos['left_y'] = y
        if(isInEyeBox("right", x, y)):
            eyes_pos['right_x'] = x
            eyes_pos['right_y'] = y
        cv2.circle(original, (x, y), 2, (255, 0 ,0))
    cv2.circle(original, (aX, aY), 2, (0, 255 ,0))
    cv2.circle(original, (bX, bY), 2, (0, 255 ,0))
    cv2.circle(original, (cX, cY), 2, (0, 0 ,255))
    cv2.circle(original, (dX, dY), 2, (0, 0 ,255))
    
    cv2.imwrite(folder + "points/" + filename + end,original)

    if None not in eyes_pos.viewvalues():
        image =  Image.open(folder + filename + end)
        CropFace(image, eye_left=(eyes_pos['left_x'],eyes_pos['left_y']), eye_right=(eyes_pos['right_x'],eyes_pos['right_y']), offset_pct=(0.25,0.3), dest_sz=(200,200)).save(folder+ "crop/" + filename + end)
    # initialize the figure
    #fig = plt.figure("Images")
    #images = ("Original", original), ("Contrast", contrast)

    # compare the images
    #compare_images(original, contrast, "Original vs. Contrast")
numFaces = 0
'''listFaces = os.listdir("faces")
for file in listFaces:
    if file.endswith(".jpg"):
        numFaces += 1
        print(str(float(numFaces) / len(listFaces) * 100) + "%")
        updateModel(file[:-4])'''
listCrop = os.listdir("faces/crop")
original = cv2.imread("faces/crop/1.jpg")
for file in listCrop:
    compare = cv2.imread("faces/crop/" + file)
    compare_images(original, compare)
