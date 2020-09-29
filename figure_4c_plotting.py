import matplotlib.pyplot as plt
import numpy as np
import h5py
import seaborn as sns


import plot_format

plot_format.set_format()

#data_file = h5py.File("figure_4c.h5", "r")
data_file = h5py.File("extra_fine_discrete.h5", "r")


key = "foo"
TE_vals = data_file[key]["TE"]
num_events = data_file[key]["num_target_events"]
dt = data_file[key]["dt"].value
print(TE_vals.shape)

fig, axs = plt.subplots(nrows = TE_vals.shape[2], figsize = (8, 15))
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

    sns.lineplot(x = num_events, y = means, palette = "Set3", linewidth = 4, ax = axs[i])
    axs[i].fill_between(num_events, means - stds, means + stds, alpha = 0.5)
    axs[i].hlines(0.5076, 0, num_events[-1], color = "black", linewidth = 3)

    axs[i].set_xscale("log")
    axs[i].set_ylim([-0.1, 1.8])
    axs[i].set_title("$\Delta t$ = " + str(dt[i]))
    #axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #axs[i].xaxis.set_major_formatter(LogFormatterSciNotation('%.2f'))



axs[1].set_ylabel("TE (nats/second)")
axs[0].set_xlabel("")
axs[1].set_xlabel("")
axs[2].set_xlabel("")
#axs[3].set_xlabel("Number of Target Events")
axs[3].set_xlabel("")
axs[4].set_xlabel("")
axs[5].set_xlabel("")
axs[6].set_xlabel("Number of Target Events")
plt.subplots_adjust(hspace = 0.4)

#plt.show()
plt.savefig("extra_fine_discrete.pdf", bbox_inches='tight', format = "pdf")
