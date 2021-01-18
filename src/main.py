#Global imports

#Local imports
import pipeline
import evaluations


if __name__ == "__main__":
    # prepare_data.print_data_stat()
    # evaluations.eval_perfomance_lbph_svm(MODE, VERBOSE)
    # evaluations.eval_pca(VERBOSE)
    # evaluations.eval_ada()
    
    VERBOSE = True
    #two modes: test-> for debugging    deliver-> final version
    MODE = 'test'
    while True:
        pipeline.pipe(clf='svm',_verbose=VERBOSE, _mode=MODE)
        