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
    # evaluations.final_eval()

    VERBOSE = False
    MODE = 'deliver'
    data_path = 'data/'
    test_folders = os.listdir(data_path)
    test_folders = sorted(test_folders,key=int)
    if os.path.exists('results.txt'):
        os.remove('results.txt')
    if os.path.exists('time.txt'):
        os.remove('time.txt')
        
    results_file = open('results.txt',"a+")
    time_file = open('time.txt',"a+")

    for test_folder in test_folders:
        res,time = pipeline.pipe(feature='cslbcop', clf='svm',_verbose=VERBOSE, _mode=MODE, test_folder=test_folder)
        time = "{:.2f}".format(time)
        results_file.write(str(res))
        results_file.write('\n')
        results_file.flush()
        time_file.write(str(time))
        time_file.write('\n')
        time_file.flush()

    results_file.close()
    time_file.close()