import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import sklearn.metrics

import seaborn as sns

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],
             'size' : 18})

plt.rc('axes', titlesize=26)
plt.rc('axes', labelsize=26)
plt.rc('xtick', labelsize=22)
plt.rc('ytick', labelsize=22)
plt.rc('figure', titlesize=26)
plt.rc('axes', linewidth=3)
plt.rc('xtick.major', width=3)
plt.rc('xtick.minor', width=3)

rc('text.latex', preamble=r'\usepackage{cmbright}')

data_file = h5py.File("figure_2.h5", "r")

for key in data_file.keys():
    TE_vals = data_file[key]["TE"]
    num_events = data_file[key]["num_target_events"]
    mu = data_file[key]["mu"].value
    print(TE_vals.shape)


    fig, axs = plt.subplots(nrows = TE_vals.shape[2], figsize = (6, 12))
    for i in range(TE_vals.shape[2]):
        means = []
        stds = []
        for j in range(TE_vals.shape[1]):
            cleaned = TE_vals[TE_vals[:, j, i] != -100, j, i]
            means.append(np.mean(cleaned))
            stds.append(np.std(cleaned))

        means = np.array(means)
        stds = np.array(stds)

        sns.lineplot(x = num_events, y = means, palette = "Set3", linewidth = 4, ax = axs[i])
        axs[i].fill_between(num_events, means - stds, means + stds, alpha = 0.5)
        axs[i].hlines(0.0, 0, num_events[-1])

        axs[i].set_xscale("log")
        axs[i].set_ylim([-0.5, 0.5])
        axs[i].set_title("$N_U/N_X$ = " + str(mu[i]))
        #plt.setp(axs[i].spines.values(), linewidth=3)

    axs[1].set_ylabel("TE (nats/second)")
    axs[0].set_xlabel("")
    axs[1].set_xlabel("")
    axs[2].set_xlabel("")
    axs[3].set_xlabel("Number of Target Events")
    plt.subplots_adjust(hspace = 0.5)

    #plt.show()
    plt.savefig("bias_at_samples_"
                + str(data_file[key]["k"].value) +".pdf",
                bbox_inches='tight', format = 'pdf')
