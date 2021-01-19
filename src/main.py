#Global imports
from global_imports import plt
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
    evaluations.final_eval(_mode=MODE, _verbose=VERBOSE)
    while True:
        res = pipeline.pipe(feature='cslbcop', clf='svm',_verbose=VERBOSE, _mode=MODE)
            

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
        
