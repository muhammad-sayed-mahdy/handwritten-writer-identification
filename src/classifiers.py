#Global imports
from global_imports import np, svm, mode, StatisticsError, KNeighborsClassifier, AdaBoostClassifier, DecisionTreeClassifier
import random
#local imports



def call_svm(X_tune, y_tune, X_test, y_test, verbose=False, _mode='test'):
    '''
        + Our first classifier. SVM (support vector machine).
        + This aims to classify components based on a marginal error, to better separate data.
        + steps: 
            0- declare
            1- fitting
            2- predict
            3- score
        + TUNEs:
            1- kernel used: between `rbf`, `linear` and `poly`, poly shows slightly better performance.
            2- C (regularization) as it turned out, on average 4000 is a very good estimate
            3- degree: this only concerns `poly` and usually we'd go for a degree of 3
                but after tuning it turned out that 1 is a better option.
        + returns:
            0/1: for wrong/correct predictions
            accuracy: a float [0,1]
    '''
    #step 0: declare
    clf = svm.SVC(kernel='poly', C=4000,gamma='scale', probability= True, degree=1, tol=1)
    #step 1: fit
    clf.fit(X_tune, y_tune)
    #step 2: predict
    y_pred = clf.predict(X_test)
    #step 3: score
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    
    print (y_pred)
    print (y_test)
    #used in evaluations and verboses.
    if _mode == 'test':
        y_pred_most = -1
        stat_error = False
        try:
            y_pred_most =  (mode(y_pred))
        except StatisticsError:
            print (f'False prediction (equal values) -- y_pred = {y_pred}\ty_test = {y_test[0]}')
            stat_error = True

        if verbose : print(f'True Author: {y_test[0]}\tPred Author: {y_pred_most}')
        if verbose : print (f"Predicted with accuracy:\t{accuracy*100}%")
    
        if y_test[0] == y_pred_most:
            confidence = (y_pred == y_pred_most).sum() / y_pred.shape[0]
            return 1, confidence, y_pred_most

        if not stat_error: #and yet its a misclf.
            print (f'False prediction -- y_pred = {y_pred}\ty_test = {y_test[0]}')

        y_rand = y_pred[0]
        confidence = (y_pred == y_rand).sum() / y_pred.shape[0]

        return 0, confidence, y_rand

    #incase of training
    if verbose : print (f"Predicted with accuracy:\t{accuracy*100}%")
    return None, accuracy

def call_adaboost(X_tune, y_tune, X_test, y_test, verbose=False, _mode='test'):
    # acrs = []
    # est_list = np.linspace(2,3.7,num=10)
    # for est in est_list:
    #     estm = int(pow(10,est))
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators = 300)
    clf.fit(X_tune, y_tune)
    y_pred = clf.predict(X_test)
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print (y_pred)
    print (y_test)
    #used in evaluations and verboses.
    if _mode == 'test':
        y_pred_most = -1
        stat_error = False
        try:
            y_pred_most =  (mode(y_pred))
        except StatisticsError:
            print (f'False prediction (equal values) -- y_pred = {y_pred}\ty_test = {y_test[0]}')
            stat_error = True

        if verbose: print(f'True Author: {y_test[0]}\tPred Author: {y_pred_most}')
        if verbose: print (f"Predicted with accuracy:\t{accuracy*100}%")
    
        if y_test[0] == y_pred_most:
            confidence = (y_pred == y_pred_most).sum() / y_pred.shape[0]
            return 1, confidence, y_pred_most

        if not stat_error: #and yet its a misclf.
            print (f'False prediction -- y_pred = {y_pred}\ty_test = {y_test[0]}')

        y_rand = y_pred[0]
        confidence = (y_pred == y_rand).sum() / y_pred.shape[0]
        return 0, confidence, y_rand

    # print (f'n_estimators: {estm}\t\t\tacc:{accuracy}')
    #     acrs.append(accuracy)
    # return acrs



def call_knn(X_tune, y_tune, X_test, y_test, verbose=False, _mode='test'):
    '''
        + Our second classifier. KNN (K-Nearest Neighbour).
        + This aims to classify components according to the nearest k samples.
        + steps: 
            0- declare
            1- fitting
            2- predict
            3- score
        + TUNEs:
            1- n_neighbors: number of k nearest neighbours
        + returns:
            0/1: for wrong/correct predictions
            accuracy: a float [0,1]
    '''
    #step 0: declare
    neigh = KNeighborsClassifier(n_neighbors=12)
    #step 1: fit
    neigh.fit(X_tune, y_tune)
    #step 2: predict
    y_pred = neigh.predict(X_test)
    #step 3: score
    accuracy = np.sum(y_pred == y_test) / len(y_test)

    #used in evaluations and verboses.
    if _mode == 'test':
        y_pred_most = -1
        stat_error = False
        try:
            y_pred_most =  (mode(y_pred))
        except StatisticsError:
            print (f'False prediction (equal values) -- y_pred = {y_pred}\ty_test = {y_test[0]}')
            stat_error = True

        if verbose : print(f'True Author: {y_test[0]}\tPred Author: {y_pred_most}')
        if verbose : print (f"Predicted with accuracy:\t{accuracy*100}%")
    
        if y_test[0] == y_pred_most:
            return 1, accuracy

        if not stat_error: #and yet its a misclf.
            print (f'False prediction -- y_pred = {y_pred}\ty_test = {y_test[0]}')

        return 0, accuracy

    #incase of training
    if verbose : print (f"Predicted with accuracy:\t{accuracy*100}%")
    return None, accuracy
