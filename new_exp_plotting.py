import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import sklearn.metrics
from matplotlib.ticker import FormatStrFormatter, LogFormatterSciNotation
import seaborn as sns
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],
             'size' : 18})

plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('figure', titlesize=18)

rc('text.latex', preamble=r'\usepackage{cmbright}')

NUM_RUNS = 30
SIZES = [8, 16, 24]
TARGET_TRAIN_LENGTHS = [int(1e2), int(5e2), int(1e3), int(2e3), int(5e3), int(1e4)]

exc_p = np.zeros((NUM_RUNS, len(SIZES), len(TARGET_TRAIN_LENGTHS)))
inh_p = np.zeros((NUM_RUNS, len(SIZES), len(TARGET_TRAIN_LENGTHS)))
fake_p = np.zeros((NUM_RUNS, len(SIZES), len(TARGET_TRAIN_LENGTHS)))
fake_corr_p = np.zeros((NUM_RUNS, len(SIZES), len(TARGET_TRAIN_LENGTHS)))

for i in range(NUM_RUNS):

    data_file = h5py.File("correlated_pop_discrete_pairwise/run_" + str(i + 1) + ".h5", "r")

    for key in data_file.keys():
        p = data_file[key]["p"].value
        net_size_index = len(SIZES) - data_file[key]["net_size"].value - 1
        extra_type = str(data_file[key]["extra_type"].value)[2:-1]
        target_length = data_file[key]["target_length"].value
        target_length_index = TARGET_TRAIN_LENGTHS.index(target_length)

        if extra_type == "exc":
            exc_p[i, net_size_index, target_length_index] = p
        elif extra_type == "inh":
            inh_p[i, net_size_index, target_length_index] = p
        elif extra_type == "fake_corr":
            fake_corr_p[i, net_size_index, target_length_index] = p
        else:
            fake_p[i, net_size_index, target_length_index] = p

#exc_p = np.mean(exc_p, axis = 0)
exc_p = exc_p < 0.05
inh_p = inh_p < 0.05
fake_p = fake_p < 0.05
fake_corr_p = fake_corr_p < 0.05
exc_p = np.sum(exc_p, axis = 0)/NUM_RUNS
inh_p = np.sum(inh_p, axis = 0)/NUM_RUNS
fake_p = np.sum(fake_p, axis = 0)/NUM_RUNS
fake_corr_p = np.sum(fake_corr_p, axis = 0)/NUM_RUNS


def make_heatmap(title, p_vals):
    sns.heatmap(p_vals, vmin = 0, vmax = 1)
    plt.title(title)
    plt.xticks(ticks = np.arange(len(TARGET_TRAIN_LENGTHS)) + 0.5, labels = TARGET_TRAIN_LENGTHS)
    plt.yticks(ticks = np.flip(np.arange(3) + 0.5), labels = SIZES)
    plt.xlabel("num target spikes")
    plt.ylabel("num conditioning processes")
    plt.show()

make_heatmap("Excitatory true positive rate", exc_p)
make_heatmap("Inhibitory true positive rate", inh_p)
make_heatmap("Uncorrelated false positive rate", fake_p)
make_heatmap("Correlated false positive rate", fake_corr_p)
