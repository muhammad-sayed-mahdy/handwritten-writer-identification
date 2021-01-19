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
    evaluations.eval_perfomance_lbph_svm(VERBOSE=VERBOSE, MODE=MODE)
        