#Global imports
from global_imports import np, svm, mode, StatisticsError, KNeighborsClassifier, AdaBoostClassifier, DecisionTreeClassifier
import random
from collections import Counter
#local imports



def score(y_pred, y_test=None, _verbose=False):
    
    confd = []
    length = y_pred.shape[0]
    
    cs = Counter(y_pred)
    best = cs.most_common(1)[0][0]

    for i in range (3):
        #0,1,2
        confd.append(cs[i]/length)


    if _verbose:
        print (y_pred)
        print (y_test)
        
    if y_test is None:
        #Deliver mode
        #returns: best result, conf_list, None
        return best, confd, None
    else:
        #test mode
        #returns: best result, conf_list, true/false
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        if _verbose:
            print(f'True Author: {y_test[0]}\tPred Author: {best}')
            print (f"Accuracy:\t{accuracy*100}%")
        return best, confd, (y_test[0]==best)


def call_svm(X_tune, y_tune, X_test, y_test, _verbose=False, _mode='test'):
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
    clf = svm.SVC(kernel='poly', C=4000,gamma='scale', probability= True, degree=1, tol=0.99)
    #step 1: fit
    clf.fit(X_tune, y_tune)
    #step 2: predict
    y_pred = clf.predict(X_test)
    #step 3: score
    if _mode == 'test':
        return score(y_pred,y_test=y_test, _verbose=_verbose)
    elif _mode == 'deliver':
        return score(y_pred)


def call_adaboost(X_tune, y_tune, X_test, y_test, _verbose=False, _mode='test'):
    # acrs = []
    # est_list = np.linspace(2,3.7,num=10)
    # for est in est_list:
    #     estm = int(pow(10,est))
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators = 300)
    clf.fit(X_tune, y_tune)
    y_pred = clf.predict(X_test)
    #step 3: score
    if _mode == 'test':
        return score(y_pred,y_test=y_test, _verbose=_verbose)
    elif _mode == 'deliver':
        return score(y_pred)

    # print (f'n_estimators: {estm}\t\t\tacc:{accuracy}')
    #     acrs.append(accuracy)
    # return acrs

def call_knn(X_tune, y_tune, X_test, y_test, _verbose=False, _mode='test'):
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
    if _mode == 'test':
        return score(y_pred,y_test=y_test, _verbose=_verbose)
    elif _mode == 'deliver':
        return score(y_pred)