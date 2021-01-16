from matplotlib import pyplot as plt
import cv2
import numpy as np
import segmentation


def preprocess(image):
    #grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #binary
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)

    xy_coord_lines = segmentation.get_lines_coord(thresh)
    list_images = []
    for xy in xy_coord_lines:
        list_images.append(thresh[xy[0]:xy[1],xy[2]:xy[3]] )
    return list_images
    
