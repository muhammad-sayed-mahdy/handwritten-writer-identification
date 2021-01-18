#Global imports

#Local imports
import pipeline
import evaluations


if __name__ == "__main__":
    # prepare_data.print_data_stat()

    VERBOSE = True
    MODE = 'test'
    # evaluations.eval_perfomance_lbph_svm(MODE, VERBOSE)
    # evaluations.eval_pca(VERBOSE)
    # evaluations.eval_ada()
    while True:
        pipeline.pipe(clf='adaboost',_verbose=VERBOSE)
        