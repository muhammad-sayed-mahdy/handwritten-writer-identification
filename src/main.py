#Global imports
from global_imports import cv2, np, plt, Image, time
# import os
#Local imports
import preprocessing
import features
import classifiers
import prepare_data
import evaluations


def preprocess_feature(paths):
    '''
        + Main Pipeline starts here.
    '''
    global VERBOSE

    X_list = []
    y_list = []
    for author_i,tr in enumerate(paths):

        for image_i,image_path in enumerate(tr):
            if VERBOSE: print (f'image\t{image_i}\tof author:\t{author_i}')
            image = cv2.imread(image_path)
            list_images = preprocessing.preprocess(image)
            for img in list_images:
                # x = Image.fromarray(img)
                # x.show()
                coeffs = features.waveletTransform(img,'db4')
                cA ,(cH,cV,cD) = coeffs
                hist_of_line_horizontal = features.LPBH(cH,1,8)
                hist_of_line_vertical = features.LPBH(cV,1,8)
                hist_of_line_diagonal = features.LPBH(cD,1,8)

                # stacking histograms(features)
                hist_of_line = hist_of_line_horizontal + hist_of_line_vertical + hist_of_line_diagonal

                X_list.append(hist_of_line)
                y_list.append(author_i)
            
    if VERBOSE: print(f'X shape:\t{len(X_list) , len(X_list[0])}')
    return np.array(X_list),np.array(y_list)
    



if __name__ == "__main__":
    # prepare_data.print_data_stat()

    VERBOSE = False
    DEBUG = False
    MODE = 'test'
    #fetch data
    tr = 0
    pre = 0
    acc = 0
    trueacc = np.zeros(101)
    falseacc = np.zeros(101)
    allacc = np.zeros(101)
    avgrtconv = 0
    for i in range(500):
        train, test = prepare_data.fetch_data(mode=MODE, debug=DEBUG)
        X_tune, y_tune = preprocess_feature(train)
        X_test, y_test = preprocess_feature(test)
        pre, acc = classifiers.call_svm(X_tune, y_tune, X_test, y_test, verbose=VERBOSE, _mode=MODE)
        tr += pre
        acc = int(100*acc + 0.5)
        allacc[acc] += 1
        if acc > 50:
            avgrtconv += acc
        if pre == 1:
            trueacc[acc] += 1
        else:
            falseacc[acc] += 1
    print(tr/5)
    print(avgrtconv/500)
    plt.plot(allacc)
    plt.show()
    plt.plot(trueacc)
    plt.show()
    plt.plot(falseacc)
    plt.show()
# accuracy = 96.2%,   average confidence in true detected writers = 86.756
