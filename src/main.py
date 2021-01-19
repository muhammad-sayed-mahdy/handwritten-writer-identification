#Global imports

#Local imports
import pipeline
import evaluations


if __name__ == "__main__":
    # prepare_data.print_data_stat()
    # evaluations.eval_perfomance_lbph_svm(MODE, VERBOSE)
    # evaluations.eval_pca(VERBOSE)
    # evaluations.eval_ada()
    
    VERBOSE = False
    #two modes: test-> for debugging    deliver-> final version
    MODE = 'test'
    trials = 0
    correct = 0
    for i in range (2000):
        res = pipeline.pipe(clf='svm',_verbose=VERBOSE, _mode=MODE)
        if res: correct += 1
        trials += 1
        print (f'Trial: {trials}\t\tCorrect: {correct}\t\tOverall: {correct/trials}')
        
