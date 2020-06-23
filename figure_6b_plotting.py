import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import sklearn.metrics

import seaborn as sns
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],
             'size' : 14})

plt.rc('axes', titlesize=14)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=12)
plt.rc('figure', titlesize=16)

rc('text.latex', preamble=r'\usepackage{cmbright}')

data_file = h5py.File("figure_6.h5", "r")

for key in data_file.keys():
    TE_vals = data_file[key]["TE"].value
    TE_surrogates = data_file[key]["TE_surrogate"].value
    TE_shift_surrogates = data_file[key]["TE_shift_surrogate"].value
    shifts = data_file[key]["shifts"].value

    mean_surrogates = np.mean(TE_surrogates, axis = 1)
    std_surrogates = np.std(TE_surrogates, axis = 1)
    mean_shift_surrogates = np.mean(TE_shift_surrogates, axis = 1)
    std_shift_surrogates = np.std(TE_shift_surrogates, axis = 1)

    corrected_good = TE_vals - mean_surrogates.reshape(-1, 1)
    corrected_shifts = TE_vals - mean_shift_surrogates.reshape(-1, 1)

    mean = np.mean(TE_vals, axis = 1)
    std = np.std(TE_vals, axis = 1)

    mean_good = np.mean(corrected_good, axis = 1)
    std_good = np.std(corrected_good, axis = 1)
    mean_shifts = np.mean(corrected_shifts, axis = 1)
    std_shifts = np.std(corrected_shifts, axis = 1)

    plt.clf()
    plt.figure(figsize=(10,5))
    #sns.lineplot(x = shifts, y = mean, palette = "Set3", linewidth = 1.5, label = "Raw TE")
    #plt.fill_between(shifts, mean - std, mean + std, alpha = 0.5)
    sns.lineplot(x = shifts, y = mean_good, palette = "Set3", linewidth = 1.5, label = "Adjusted by\nPermutation\nSurrogate")
    plt.fill_between(shifts, mean_good - std_good, mean_good + std_good, alpha = 0.5)
    sns.lineplot(x = shifts, y = mean_shifts, palette = "Set3", linewidth = 1.5, label = "Adjusted by\nShifted\nSurrogate")
    plt.fill_between(shifts, mean_shifts - std_shifts, mean_shifts + std_shifts, alpha = 0.5)

    plt.xlabel("shift")
    plt.ylabel("TE(nats/second)")
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

    plt.tight_layout()
    plt.show()

    plt.savefig("shifts_adjusted", dpi = 1000)
