#global imports
from global_imports import cv2, np

#local imports
import features
import preprocessing
import prepare_data
import classifiers

def step_0(_mode):
    '''
        + Fetches random images for either training or testing.
    '''
    return prepare_data.fetch_data(mode=_mode)
        

def step_1(paths, VERBOSE=False):
    '''
        + Main Pipeline starts here.
        + Takes a path to tune or test data and starts fetching, segmentation and features extraction.
        + Resulting X with rows representing lines and columns representing features.
        + and y vector that represent id of an author (0/1/2)
    '''
    
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

def step_2(X_tune, y_tune, X_test, y_test, verbose=False, _mode='test'):
   return classifiers.call_svm(X_tune, y_tune, X_test, y_test, verbose=verbose, _mode=_mode)


def pipe(feature='lbph', clf='svm', _mode='test', _verbose=False):
    '''
        + This is the main function call for this file.
        + It specifies which feature ext. technique and which classifier to be used.
    '''
    train, test = step_0(_mode=_mode)
    X_tune, y_tune = step_1(train,_verbose)
    X_test, y_test = step_1(test,_verbose)
    pre, acc = step_2(X_tune, y_tune, X_test, y_test, verbose=_verbose, _mode=_mode)
    return pre, acc