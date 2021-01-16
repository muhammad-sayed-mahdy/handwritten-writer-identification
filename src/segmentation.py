import cv2
import numpy as np

def discard(image):
    image = np.uint8(image)
    _, im_label, stts, _ = cv2.connectedComponentsWithStats(image, connectivity=4)
    msk = np.isin(im_label, np.where(stts[:, cv2.CC_STAT_WIDTH] > 500)[0])
    image[msk] = 0
    return image



def get_lines_coord(image):
    # Discarding long connected components
    image_without_lines = discard(image.copy())
    image_with_just_lines = cv2.bitwise_xor(image, image_without_lines)
    sum_horizontal = np.sum(image_with_just_lines,axis=1)

    lines_indexes = np.argwhere(sum_horizontal>2000).flatten()
    y_start = lines_indexes[ np.argmax(lines_indexes > (lines_indexes[0]+150) ) ]
    y_end = max(lines_indexes)

    kernel = np.ones((5,100), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=2)
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    lines_of_text = []
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding boxl
        x, y, w, h = cv2.boundingRect(ctr)
        if(y< y_end and y>y_start and h>80):
            lines_of_text.append((y,y+h,x,x+w))
    return  lines_of_text