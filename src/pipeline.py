#global imports
from global_imports import cv2, np, PCA, StandardScaler
import time
#local imports
import prepare_data     #step 0
import preprocessing    #step 1.1
import features         #step 1.2
import classifiers      #step 2
import evaluations

def step_0(_mode = 'test', _verbose=False):
    '''
        + Fetches random images for either training or testing.
    '''
    if _mode == 'test' or _mode == 'deliver':
        train_paths, test_paths = prepare_data.fetch_data(_mode=_mode)
        train_images, test_images = [],[]
        test_label = None
        #READ TRAIN
        for author_i, tr in enumerate(train_paths):
            for _, image_path in enumerate(tr):
                image = cv2.imread(image_path)
                train_images.append(image)
        #READ TEST
        for author_i, tr in enumerate(test_paths):
            if len(tr) > 0:
                image = cv2.imread(tr[0])
                test_images.append(image)
                test_label = author_i
    
        return train_images, test_images, test_label
    # elif _mode == 'deliver':
    #     pass
        

def step_1(images, VERBOSE=False, test_label=None):
    '''
        + Main Pipeline starts here.
        + Takes a path to tune or test data and starts fetching, segmentation and features extraction.
        + Resulting X with rows representing lines and columns representing features.
        + and y vector that represent id of an author (0/1/2)
    '''
    
    X_list = []
    y_list = []
    for author_i, image in enumerate(images):
        lines_count = 0
        list_images = preprocessing.preprocess(image)
        for img in list_images:
            # x = Image.fromarray(img)
            # x.show()
            coeffs = features.waveletTransform(img,'db4')
            cA ,(cH,cV,cD) = coeffs
            hist_of_line = features.LPBH(cA,1,8)
            #hist_of_line_horizontal = features.LPBH(cH,1,8)
            #hist_of_line_vertical = features.LPBH(cV,1,8)
            #hist_of_line_diagonal = features.LPBH(cD,1,8)

            # stacking histograms(features)
            #hist_of_line = np.hstack((hist_of_line_horizontal, hist_of_line_vertical, hist_of_line_diagonal))
            lines_count += 1
            X_list.append(hist_of_line)
            if test_label is None:
                y_list.append(author_i//2)
            else:
                y_list.append(test_label)
        if VERBOSE: print (f'Author: {author_i//2}\t #lines: {lines_count}')
    if VERBOSE: print(f'X shape:\t{len(X_list) , len(X_list[0])}')
    return np.array(X_list), np.array(y_list)

def step_2(X_tune, X_test, y_tune,_verbose=False,n_components=33):
    #scale
    sc = StandardScaler()
    X_tune = sc.fit_transform(X_tune)
    X_test = sc.transform(X_test)
    #pca
    n_components = min(33,X_tune.shape[0]-1)
    if _verbose: print (f'N_cmp: {n_components}')
        
    pca = PCA(n_components=n_components, copy=False)
    X_tune = pca.fit_transform(X_tune)
    X_test = pca.transform(X_test)

    if _verbose: print (f'New Shapes: X_tune: {X_tune.shape}\ty_tune: {y_tune.shape}')
    # if _verbose: print(pca.explained_variance_ratio_)
    # if _verbose: print(pca.singular_values_)
    return X_tune, X_test

def step_3(X_tune, y_tune, X_test, y_test, _verbose=False, _mode='test', clf='svm'):
    '''
        + For now it's just a calling function, but later on, this will be passed the classifier technique
        used. and choose from it.
    '''
    if clf == 'svm':
        return classifiers.call_svm(X_tune, y_tune, X_test, y_test, 
                    _verbose=_verbose, _mode=_mode)
    elif clf == 'adaboost':
        return classifiers.call_adaboost(X_tune, y_tune, X_test, y_test, 
                    _verbose=_verbose, _mode=_mode)
    elif clf == 'knn':
        return classifiers.call_knn(X_tune, y_tune, X_test, y_test, 
                    _verbose=_verbose, _mode=_mode)
         


def pipe(feature='lbph', clf='svm', _mode='test', 
            _verbose=False,pca_scatter=False,n_components=33):
    '''
        + This is the main function call for this file.
        + It specifies which feature ext. technique and which classifier to be used.
    '''
    if _verbose: print ('\n\t\tFetch..')
    
    train_images, test_images, test_label = step_0(_mode=_mode)
    
    if _verbose: print ('\t\tPreprocess and FE..')
    start_time = time.time()
    
    X_tune,y_tune = step_1(train_images,_verbose)
    X_test,y_test = step_1(test_images,_verbose,test_label=test_label)
    
    if _verbose: print ('\t\tPCA..')
    
    X_tune,X_test = step_2(X_tune,X_test,y_tune, _verbose=_verbose,n_components=n_components)
    
    if _verbose: print ('\t\tCLF..')
    
    #best_result, conf_list, correct
    if _mode == 'deliver': #TODO:CAHNGE IT
        print(f"using {clf} --- {(time.time() - start_time)} seconds ---")
        return step_3(X_tune, y_tune, X_test, y_test,_verbose=_verbose, _mode=_mode, clf=clf)
    else:
        #deliver
        best_svm, conf_svm, _ = step_3(X_tune, y_tune, X_test, y_test=None,_verbose=_verbose, _mode=_mode, clf='svm')
        if max(conf_svm) != 1.0:
            #try ada and knn
            best_ada, conf_ada, _ = step_3(X_tune, y_tune, X_test, y_test=None,_verbose=_verbose, _mode=_mode, clf='adaboost')
            best_knn, conf_knn, _ = step_3(X_tune, y_tune, X_test, y_test=None,_verbose=_verbose, _mode=_mode, clf='knn')
            confds  = [a + b + c for a, b, c in zip(conf_svm, conf_ada, conf_knn)]
            best = None
            max_val = -1
            for i,c in enumerate(confds):
                if c > max_val:
                    max_val = c
                    best = i
            print(f"--- {(time.time() - start_time)} seconds ---")
            if _verbose: print (f'PICKED {best}')
            return (best==y_test[0])
        else:
            print(f"--- {(time.time() - start_time)} seconds ---")
            if _verbose: print (f'PICKED {best_svm}')
            return (best_svm==y_test[0])
    