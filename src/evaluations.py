#global imports
from global_imports import plt, np, Image, cv2
import random, os
#local imports
import pipeline

def plot_regz_acrs(accrs=None):
    ys = np.linspace(0,2, num=3).astype('int')
    # ys = np.power(10,ys).astype('int')
    print (ys)
    plt.bar(ys,accrs)
    plt.xticks(ys, ('linear','poly','rbf'))
    plt.show()


def eval_perfomance_lbph_svm(MODE, VERBOSE=False):
    #fetch data
    tr = 0
    pre = 0
    conf = 0
    trueacc = np.zeros(101)
    falseacc = np.zeros(101)
    allacc = np.zeros(101)
    avgrtconv = 0
    predicted =-1
    cnt = 0
    for i in range(100):
        print("Iteration: ", i+1)
        predicted, conf, pre = pipeline.pipe(feature='lbph', clf='svm',_mode='test', _verbose=VERBOSE)
        conf = int(100*conf[predicted] + 0.5)
        allacc[conf] += 1
        if pre == 1:
            trueacc[conf] += 1
        else:
            falseacc[conf] += 1
        if pre == 1 and conf > 50:
            tr += 1
            avgrtconv += conf
        if pre == 0 and conf > 50:
            cnt += 1
    print(tr)
    print(avgrtconv/100)
    print(cnt)
    plt.plot(allacc)
    plt.show()
    plt.plot(trueacc)
    plt.show()
    plt.plot(falseacc)
    plt.show()
# accuracy = 96.2%,   average confidence in true detected writers = 86.


def plot_scatter_pca(X_tune, y_tune):
    # x, y = np.random.rand(2, N)
    # c = np.random.randint(1, 5, size=N)
    # s = np.random.randint(10, 220, size=N)

    fig, ax = plt.subplots()
    scatter = ax.scatter(X_tune[:,0],X_tune[:,1], c =y_tune, s=50)

    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower left", title="Classes")
    ax.add_artist(legend1)
    # plt.show()
    name= 'graphs/pca/failed/pca_'+str(random.randint(0,100))+'.png'
    plt.savefig(name)
    plt.clf()

def eval_pca(VERBOSE=True):
    trials = 0
    total_correct = 0
    n_comp = 50
    for _ in range(2000):
        pre, conf, correct = pipeline.pipe(feature='lbph', clf='svm',_mode='test', 
            _verbose=VERBOSE, pca_scatter=False, n_components=n_comp)
        total_correct += correct
        trials += 1
        print (f'Trial: {trials}\tCorrects: {total_correct}\tConf: {conf}\tOverall: {total_correct/trials}')
        

def eval_ada(VERBOSE=True):
    
    acr = np.zeros(10)
    for _ in range (100):
        _, _, correct = pipeline.pipe(feature='lbph', clf='adaboost',_mode='test', 
            _verbose=VERBOSE)
        acr += (correct)
    
    est_list = np.linspace(2,3.7,num=10)
    plt.bar(est_list,acr)
    plt.show()
    plt.savefig('graphs/ada_n_estimators.png')
    

def show_damage( train_paths, test_paths):
    '''
        FOR SURE we've failed.
    '''
    print (train_paths)
    print (test_paths)
    
    rows = 3
    cols = 3
    axes=[]
    fig=plt.figure(figsize=(16, 12), dpi=80)
    i = 0
    for author_i, author_files in enumerate(train_paths):
        for _,tr in enumerate(author_files):
            i += 1
            b = cv2.imread(tr)
            axes.append( fig.add_subplot(rows, cols, i) )
            subplot_title=("author"+str(author_i))
            axes[-1].set_title(subplot_title)  
            plt.imshow(b)
    
    for author_i, author_files in enumerate(test_paths):
        for _,tr in enumerate(author_files):
            i += 1
            b = cv2.imread(tr)
            axes.append( fig.add_subplot(rows, cols, i) )
            subplot_title=("author"+str(author_i))
            axes[-1].set_title(subplot_title)  
            plt.imshow(b)

    # fig.tight_layout()    
    # plt.show()


            

    input ("PROCEED?")



def final_eval(_mode='test', _verbose=True):
    itr = 0
    failed = 0
    set_id = 0
    data_path = 'data/' 
    datas = os.listdir(data_path)
    for set_id in range(159):
        auth = datas[set_id]
        form_count = len(os.listdir(data_path+auth))
        forms = os.listdir(data_path+auth)
        for form_id in range(form_count):
            itr += 1
            res = pipeline.pipe(feature='cslbcop', clf='svm',_verbose=_verbose, _mode=_mode,set_id=set_id, form_id=form_id)
            if not res: failed += 1
            print(f'Trial: {itr}\tFailed: {failed}\tauthor: {set_id} \tform: {form_id}\n\n')
            
