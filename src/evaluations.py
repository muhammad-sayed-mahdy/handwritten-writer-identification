#global imports
from global_imports import plt, np



def plot_regz_acrs(accrs=None):
    ys = np.linspace(2,4, num=15)
    ys = np.power(10,ys).astype('int')
    plt.plot(ys,accrs)
    plt.show()
