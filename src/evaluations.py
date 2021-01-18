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
    acc = 0
    trueacc = np.zeros(101)
    falseacc = np.zeros(101)
    allacc = np.zeros(101)
    avgrtconv = 0
    for i in range(500):
        pre, acc = pipeline.pipe(feature='lbph', clf='svm',_mode='test', _verbose=VERBOSE)
        tr += pre
        acc = int(100*acc + 0.5)
        allacc[acc] += 1
        if acc > 50:
            avgrtconv += acc
        if pre == 1:
            trueacc[acc] += 1
        else:
            falseacc[acc] += 1
    print(tr/5)
    print(avgrtconv/500)
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
    scatter = ax.scatter(X_tune[:,0],X_tune[:,1], c =y_tune)

    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower left", title="Classes")
    ax.add_artist(legend1)

    # produce a legend with a cross section of sizes from the scatter
    # handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
    # legend2 = ax.legend(handles, labels, loc="upper right", title="Sizes")

    plt.show()
