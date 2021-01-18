#global imports
from global_imports import plt, np

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
    cnt = 0
    for i in range(100):
        print("Iteration: ", i+1)
        pre, conf = pipeline.pipe(feature='lbph', clf='svm',_mode='test', _verbose=VERBOSE)
        conf = int(100*conf + 0.5)
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
# accuracy = 96.2%,   average confidence in true detected writers = 86.756