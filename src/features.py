#Global imports
from global_imports import np, pywt, cv2, feature, Image

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

def LPB(img,rad,p=8,xgrid =8 ,ygrid =8):
    img_lbp = feature.local_binary_pattern(img,p,rad)

    # should divide img into histogram grids and return histograms
    xdim = img_lbp.shape[0]
    ydim = img_lbp.shape[1]
    xstep = xdim//xgrid
    ystep  = ydim//ygrid
    list_histograms = []
    
    img_lbp = np.uint8(img_lbp)
    # x = Image.fromarray(img_lbp)
    # x.show()

    for i in range(xgrid):
        xstart = 0
        ystart = 0
        for j in range(ygrid):
            # x = Image.fromarray(img_lbp[xstart:xstart+xstep,ystart:ystart+ystep])
            # x.show()
            histogram = cv2.calcHist([img_lbp[xstart:xstart+xstep,ystart:ystart+ystep]],[0],None,[256],[0,256])
            for h in histogram:
                list_histograms.append(h[0])
            xstart +=xstep
            ystart +=ystep
    return list_histograms