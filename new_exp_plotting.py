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
TARGET_TRAIN_LENGTHS = [int(1e2), int(5e2), int(1e3)]

exc_p = np.zeros((NUM_RUNS, len(SIZES), len(TARGET_TRAIN_LENGTHS)))
inh_p = np.zeros((NUM_RUNS, len(SIZES), len(TARGET_TRAIN_LENGTHS)))
fake_p = np.zeros((NUM_RUNS, len(SIZES), len(TARGET_TRAIN_LENGTHS)))

for i in range(NUM_RUNS):

    data_file = h5py.File("correlated_pop/run_" + str(i + 1) + ".h5", "r")

    for key in data_file.keys():
        p = data_file[key]["p"].value
        net_size_index = data_file[key]["net_size"].value
        extra_type = str(data_file[key]["extra_type"].value)[2:-1]
        target_length = data_file[key]["target_length"].value
        target_length_index = TARGET_TRAIN_LENGTHS.index(target_length)

        if extra_type == "exc":
            exc_p[i, net_size_index, target_length_index] = p
        elif extra_type == "inh":
            inh_p[i, net_size_index, target_length_index] = p
        else:
            fake_p[i, net_size_index, target_length_index] = p

#exc_p = np.mean(exc_p, axis = 0)
exc_p = exc_p < 0.05
inh_p = inh_p < 0.05
fake_p = fake_p > 0.05
exc_p = np.sum(exc_p, axis = 0)/NUM_RUNS
inh_p = np.sum(inh_p, axis = 0)/NUM_RUNS
fake_p = np.sum(fake_p, axis = 0)/NUM_RUNS


sns.heatmap(exc_p, vmin = 0, vmax = 1)
plt.title("Excitatory true positive rate")
plt.xticks(ticks = list(range(len(TARGET_TRAIN_LENGTHS))), labels = TARGET_TRAIN_LENGTHS)
plt.yticks(ticks = [0, 1, 2], labels = SIZES)
plt.xlabel("num target spikes")
plt.ylabel("num conditioning processes")
plt.show()

sns.heatmap(inh_p, vmin = 0, vmax = 1)
plt.title("Inhibitory true positive rate")
plt.xticks(ticks = list(range(len(TARGET_TRAIN_LENGTHS))), labels = TARGET_TRAIN_LENGTHS)
plt.yticks(ticks = [0, 1, 2], labels = SIZES)
plt.xlabel("num target spikes")
plt.ylabel("num conditioning processes")
plt.show()

sns.heatmap(fake_p, vmin = 0, vmax = 1)
plt.title("True negative rate")
plt.xticks(ticks = list(range(len(TARGET_TRAIN_LENGTHS))), labels = TARGET_TRAIN_LENGTHS)
plt.yticks(ticks = [0, 1, 2], labels = SIZES)
plt.xlabel("num target spikes")
plt.ylabel("num conditioning processes")
plt.show()
