#Global imports
from global_imports import np, svm
#local imports



def call_svm(X_tune, y_tune, X_test, y_test, verbose=False):


    #avg acc: 
    # regz = 100 -> 1000 -> 10K
    # reg_list = np.linspace(2,4, num=15)
    # accuracys = []
    # for reg in reg_list:
    # regz = int(pow(10,reg))

    clf = svm.SVC(kernel="poly", C=3600, probability= True, degree=3, tol=0.8)

    clf.fit(X_tune, y_tune)

    y_pred = clf.predict(X_test)

    accuracy = np.sum(y_pred == y_test) / len(y_test)
    if verbose : print (f"Total accuracy:\t{accuracy*100}%")

