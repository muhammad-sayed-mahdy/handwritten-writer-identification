#Global imports
from global_imports import np, svm, mode, StatisticsError
#local imports



def call_svm(X_tune, y_tune, X_test, y_test, verbose=False, _mode='test'):


    #avg acc: 
    # regz = 100 -> 1000 -> 10K
    # reg_list = np.linspace(0,9, num=10)
    # accuracys = []
    # for reg in reg_list:
    #     regz = int(pow(10,reg))

    clf = svm.SVC(kernel="poly", C=2500,gamma='scale', probability= True, degree=3, tol=1)

    clf.fit(X_tune, y_tune)

    y_pred = clf.predict(X_test)
    
    if _mode == 'test':
        y_pred_most = -1
        stat_error = False
        try:
            y_pred_most =  (mode(y_pred))
        except StatisticsError:
            print (f'False prediction -- y_pred = {y_pred}\ty_test = {y_test[0]}')
            stat_error = True
        if verbose : print(f'True Author:{y_test[0]}\tPred Author:{y_pred_most}')

    accuracy = np.sum(y_pred == y_test) / len(y_test)

    if verbose : print (f"Predicted with accuracy:\t{accuracy*100}%")

    if _mode == 'test' and y_test[0] == y_pred_most:
        return 1

    if _mode == 'test' and  not stat_error: #and yet its a misclf.
        print (f'False prediction -- y_pred = {y_pred}\ty_test = {y_test[0]}')
    return 0

