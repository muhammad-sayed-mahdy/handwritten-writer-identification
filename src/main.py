#Global imports

#Local imports
import pipeline
import evaluations


if __name__ == "__main__":
    # prepare_data.print_data_stat()

    VERBOSE = False
    MODE = 'test'
    # evaluations.eval_perfomance_lbph_svm(MODE, VERBOSE)
    # evaluations.eval_pca(VERBOSE)
    # evaluations.eval_ada()
    trials = 0
    correct = 0
    con = 0
    for i in range(1500):
        _,av,c = pipeline.pipe(clf='adaboost',_verbose=VERBOSE)
        trials += 1
        correct += c
        con += av
        print (f'\t\tit: {trials}\t\tCorrect: {correct}\t\tAVG Conf. {con/trials}\t\tOverall: {correct/trials}')
    