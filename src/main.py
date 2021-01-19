#Global imports
import os
#Local imports
import pipeline
import evaluations


if __name__ == "__main__":
    # prepare_data.print_data_stat()
    # evaluations.eval_perfomance_lbph_svm(MODE, VERBOSE)
    # evaluations.eval_ada()
    # evaluations.eval_pca(VERBOSE)
    # while True:
    #     evaluations.seperable()
    
    VERBOSE = False
    MODE = 'deliver'
    data_path = 'data/'
    test_folders = os.listdir(data_path)
    test_folders = sorted(test_folders,key=int)
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