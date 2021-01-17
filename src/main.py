from matplotlib import pyplot as plt
import cv2
import numpy as np
import preprocessing
import features




#import image
image = cv2.imread('input.png')
list_images = preprocessing.preprocess(image)
# for every image of line text
for img in list_images:
    img_wt = features.waveletTransform(img,'db1')
    list_histogram = features.LPB(img_wt,1,8)
    