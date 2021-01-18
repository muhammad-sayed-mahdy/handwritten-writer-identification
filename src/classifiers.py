#Global imports
from global_imports import np, svm, mode, StatisticsError
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