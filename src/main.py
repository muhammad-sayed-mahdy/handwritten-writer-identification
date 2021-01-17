#Global imports
from global_imports import cv2, np, plt
#Local imports
import preprocessing
import features

#import image
image = cv2.imread('input.png')
list_images = preprocessing.preprocess(image)
# for every image of line text
for img in list_images:
    img_wt = features.waveletTransform(img,'db1')
    list_histogram = features.LPB(img_wt,1,8)
    