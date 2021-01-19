#Global imports
from global_imports import plt, cv2, np
#Local imports


def preprocess(image):
    #grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #binary
    # ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
    thresh = 255-thresh

    xy_coord_lines = get_lines_coord(thresh)
    list_images = []
    for xy in xy_coord_lines:
        list_images.append(thresh[xy[0]:xy[1],xy[2]:xy[3]] )
    return list_images
    

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

    # closing
    kernel_closing = np.ones((5,200), np.uint8)
    img_closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_closing)
    # opening
    kernel_opening = np.ones((1,200), np.uint8)
    img_opening = cv2.morphologyEx(img_closing, cv2.MORPH_OPEN, kernel_opening)

    ctrs, hier = cv2.findContours(img_opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    lines_of_text = []
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding boxl
        x, y, w, h = cv2.boundingRect(ctr)
        if(y< y_end and y>y_start and h>25):
            lines_of_text.append((y,y+h,x,x+w))
    return  lines_of_text