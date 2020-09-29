import matplotlib.pyplot as plt
from matplotlib import rc

def set_format():
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],
        'size' : 18})
    plt.rc('axes', titlesize=26)
    plt.rc('axes', labelsize=26)
    plt.rc('xtick', labelsize=22)
    plt.rc('ytick', labelsize=22)
    plt.rc('figure', titlesize=26)
    plt.rc('axes', linewidth=3)
    plt.rc('xtick.major', width=3)
    plt.rc('xtick.minor', width=3)
    plt.rc('legend', fontsize=18)
    plt.rc('legend', handlelength=0.5)
    plt.rc('legend', borderpad=0.15)
    plt.rc('text.latex', preamble=r'\usepackage{cmbright}')
