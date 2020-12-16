import matplotlib.pyplot as plt
import numpy as np
import h5py
import seaborn as sns


import plot_format

plot_format.set_format()

data_file = h5py.File("figure_4c.h5", "r")
data_file_cont = h5py.File("figure_4_specialmain.h5", "r")
#data_file = h5py.File("extra_fine_discrete.h5", "r")


key = "foo"
TE_vals = data_file[key]["TE"]
num_events = data_file[key]["num_target_events"]
dt = data_file[key]["dt"].value

key = "bar"
TE_vals_cont = data_file_cont[key]["TE"]


fig, axs = plt.subplots(nrows = TE_vals.shape[2], figsize = (8, 16))
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

    means_cont = []
    stds_cont = []
    for j in range(TE_vals.shape[1]):
        cleaned = TE_vals_cont[TE_vals_cont[:, j, i] != -100, j, i]
        means_cont.append(np.mean(cleaned))
        stds_cont.append(np.std(cleaned))

    means_cont = np.array(means_cont)
    stds_cont = np.array(stds_cont)

    sns.lineplot(x = num_events, y = means, palette = "Set3", linewidth = 4, ax = axs[i])
    axs[i].fill_between(num_events, means - stds, means + stds, alpha = 0.5)
    sns.lineplot(x = num_events, y = means_cont, palette = "Set3", linewidth = 4, ax = axs[i])
    axs[i].fill_between(num_events, means_cont - stds_cont, means_cont + stds_cont, alpha = 0.5)
    axs[i].hlines(0.5076, 0, num_events[-1], color = "black", linewidth = 3)
    axs[i].hlines(0.0, 0.0, num_events[-1], color = "black", linewidth = 2, linestyle='--')

    axs[i].set_xscale("log")
    axs[i].set_ylim([-0.1, 1.8])
    axs[i].set_title("$\Delta t$ = " + str(dt[i]))
    #axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #axs[i].xaxis.set_major_formatter(LogFormatterSciNotation('%.2f'))



axs[3].set_ylabel("TE (nats/second)")
axs[0].set_xlabel("")
axs[1].set_xlabel("")
axs[2].set_xlabel("")
#axs[3].set_xlabel("Number of Target Events")
axs[3].set_xlabel("")
axs[4].set_xlabel("")
#axs[5].set_xlabel("")
axs[5].set_xlabel("Number of Target Events")
plt.subplots_adjust(hspace = 0.4)

#plt.show()
plt.savefig("discrete_v_continuous.pdf", bbox_inches='tight', format = "pdf")
