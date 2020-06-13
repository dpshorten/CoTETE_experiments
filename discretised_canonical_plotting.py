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

data_file = h5py.File("run_outputs/canonical_discrete.h5", "r")

for key in data_file.keys():
    TE_vals = data_file[key]["TE"]
    num_events = data_file[key]["num_target_events"]
    dt = data_file[key]["dt"].value
    print(TE_vals.shape)

    fig, axs = plt.subplots(nrows = TE_vals.shape[2], figsize = (6, 12))
    plt.tight_layout()
    for i in range(TE_vals.shape[2]):
        means = []
        stds = []
        for j in range(TE_vals.shape[1]):
            cleaned = TE_vals[TE_vals[:, j, i] != -100, j, i]
            means.append(np.mean(cleaned))
            stds.append(np.std(cleaned))

        means = np.array(means)
        stds = np.array(stds)

        sns.lineplot(x = num_events, y = means, palette = "Set3", linewidth = 2, ax = axs[i])
        axs[i].fill_between(num_events, means - stds, means + stds, alpha = 0.5)
        axs[i].hlines(0.5076, 0, num_events[-1])

        axs[i].set_xscale("log")
        axs[i].set_ylim([0.0, 1.8])
        axs[i].set_title("$\Delta t$ = " + str(dt[i]))
        #axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        #axs[i].xaxis.set_major_formatter(LogFormatterSciNotation('%.2f'))



    axs[1].set_ylabel("TE (nats/second)")
    axs[0].set_xlabel("")
    axs[1].set_xlabel("")
    axs[2].set_xlabel("")
    axs[3].set_xlabel("Number of Target Events")

    plt.savefig("figures/discrete_canonical", bbox_inches='tight')
