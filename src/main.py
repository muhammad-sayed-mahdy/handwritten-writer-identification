#Global imports
from global_imports import plt
import os
#Local imports
import pipeline
import evaluations


if __name__ == "__main__":
    # prepare_data.print_data_stat()
    # evaluations.eval_perfomance_lbph_svm(MODE, VERBOSE)
    # evaluations.eval_ada()
    # evaluations.eval_pca(VERBOSE)
    # evaluations.final_eval(_mode=MODE, _verbose=VERBOSE)
    
    VERBOSE = True
    MODE = 'deliver'
    data_path = 'data/'
    test_folders = os.listdir(data_path)
    test_folders = sorted(test_folders)
    if os.path.exists('results.txt'):
        os.remove('results.txt')
    if os.path.exists('time.txt'):
        os.remove('time.txt')
        
    for test_folder in test_folders:
        res,time = pipeline.pipe(feature='cslbcop', clf='svm',_verbose=VERBOSE, _mode=MODE, test_folder=test_folder)
        time = "{:.2f}".format(time)

        f = open('results.txt',"a+")
        f.write(str(res))
        f.write('\n')
        f.close()
        f = open('time.txt',"a+")
        f.write(str(time))
        f.write('\n')
        f.close()
        
    
    # while True:
    #     res = pipeline.pipe(feature='cslbcop', clf='svm',_verbose=VERBOSE, _mode=MODE)
            

    # trials = 0
    # correct = 0
    # dicy = {}
    # for i in range (2000):
    #     print (f'itr: {i}')
    #     if dicy.get(tot) is None:
    #         dicy[tot] = 1
    #     else:
    #         dicy[tot] += 1
    
    # plt.bar(range(len(dicy)), list(dicy.values()), align='center')
    # plt.xticks(range(len(dicy)), list(dicy.keys()))
    # plt.show()
    # plt.savefig('ev.pmg')
        # if not res: correct += 1
        # trials += 1
        # print (f'Trial: {trials}\t\tFailed: {correct}\t\tOverall: {correct/trials}')
        
