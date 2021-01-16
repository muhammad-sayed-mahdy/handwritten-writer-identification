import numpy as np
import pywt
import cv2
from skimage import feature    

def waveletTransform(imArray, mode='haar'):

    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0 

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H =  np.uint8(imArray_H)
    return imArray_H

def LPB(img,rad,p=8):
    img_lbp = feature.local_binary_pattern(img,p,rad)

    # should divide img into histogram grids and return histograms

    return img_lbp