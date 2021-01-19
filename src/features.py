#Global imports
from global_imports import np, pywt, cv2, feature, Image


# convert transform image into wavelet transform
# returns (approx coeff [low freq]- horizontal detailed coeff - vertical detailed coeff -diagonal detailed coeff [high freq])
def waveletTransform(imArray, Wname='db4'):

    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255
    # compute coefficients 
    coeffs=pywt.dwt2(imArray,Wname,mode = 'periodization')

    # cA ,(cH,cV,cD) = coeffs

    return coeffs

def LPBH(img, rad, p=8):
    img_lbp = feature.local_binary_pattern(img,p,rad)
    img_lbp = np.uint8(img_lbp)
    histogram = cv2.calcHist([img_lbp],[0],None,[256],[0,256])
    return histogram.flatten()

def CSLBCoP(img):
    img_lbp = feature.local_binary_pattern(img, 4, 1)
    img_lbp = np.uint8(img_lbp)
    glcm = feature.greycomatrix(img_lbp, [1], [0, 45, 90, 135], 16)
    return glcm.flatten()


def LTPH(img, thres = 5, xgrid = 1, ygrid = 8):
    dx = [0, 1, 1, 1, 0, -1, -1, -1]
    dy = [-1, -1, 0, 1, 1, 1, 0, -1]
    img_ltp = np.zeros_like(img, np.uint16)
    for x in range(1, len(img)-1):
        for y in range(1, len(img[0])-1):
            two = 1
            num = 0
            for i in range(8):
                nx = x + dx[i]
                ny = y + dy[i]
                add = 0
                if img[nx][ny] >= img[x][y] + thres:
                    add = 1
                elif img[nx][ny] < img[x][y] - thres:
                    add = -1
                num += add*two
                two <<= 1
            img_ltp[x][y] = num+255

    xstep = len(img_ltp)//xgrid
    ystep  = len(img_ltp[0])//ygrid
    list_histograms = []

    for i in range(xgrid):
        xstart = 0
        ystart = 0
        for j in range(ygrid):
            # x = Image.fromarray(img_lbp[xstart:xstart+xstep,ystart:ystart+ystep])
            # x.show()
            histogram = cv2.calcHist([img_ltp[xstart:xstart+xstep,ystart:ystart+ystep]],[0],None,[512],[0,512])
            for h in histogram:
                list_histograms.append(int(h[0]))
            xstart +=xstep
            ystart +=ystep
    return list_histograms
            
def freq_hist(img, rad = 1, p=8):
    img_freq = np.log(np.abs(np.fft.fft2(a=img)))
    img_freq /= 10
    img_freq[img_freq > 1] = 1
    img_freq *= 127
    img_freq = np.uint8(img_freq)
    histogram = cv2.calcHist([img_freq],[0],None,[128],[0,128])
    return histogram.flatten()
