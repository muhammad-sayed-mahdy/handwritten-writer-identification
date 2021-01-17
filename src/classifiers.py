#Global imports
from global_imports import np, svm
#local imports



def call_svm(X_tune, y_tune, X_test, y_test):

    regz = 100
    gamma = 0.1
    clf = svm.SVC(kernel="rbf", gamma=gamma, C=regz, probability= True)
    print ("Start classifying, This may take a while.....")
    clf.fit(X_tune, y_tune)
    print ("done fitting, let's test")

    y_pred = clf.predict(X_test)

    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print (f"Total accuracy: {accuracy}")