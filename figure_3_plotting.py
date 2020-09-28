import matplotlib.pyplot as plt
import numpy as np
import h5py
import seaborn as sns


import plot_format

plot_format.set_format()

data_file = h5py.File("figure_3.h5", "r")

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

        sns.lineplot(x = np.array(num_events), y = means, palette = "Set3", linewidth = 4, ax = axs[i])
        axs[i].fill_between(num_events, means - stds, means + stds, alpha = 0.5)
        axs[i].hlines(0.0, 0, num_events[-1], color = "black")

        axs[i].set_xscale("log")
        if data_file[key]["HL"].value == 2:
            axs[i].set_ylim([-0.01, 0.1])
        else:
            axs[i].set_ylim([-0.1, 1.1])
        axs[i].set_title("$\Delta t$ = " + str(dt[i]))
        #axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        #axs[i].xaxis.set_major_formatter(LogFormatterSciNotation('%.2f'))



    axs[1].set_ylabel("TE (nats/second)")
    axs[0].set_xlabel("")
    axs[1].set_xlabel("")
    axs[2].set_xlabel("")
    axs[3].set_xlabel("Number of Target Events")
    plt.subplots_adjust(hspace = 0.5)

    #plt.show()
    plt.savefig("discrete_bias_hist_" + str(data_file[key]["HL"].value) + ".pdf", bbox_inches='tight', format = "pdf")
