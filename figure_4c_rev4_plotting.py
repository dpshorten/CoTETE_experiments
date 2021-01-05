import matplotlib.pyplot as plt
import numpy as np
import h5py
import seaborn as sns


import plot_format

HISTOGRAM_OF_FREQS_UPPER = 10

plot_format.set_format()

data_file = h5py.File("figure_4c_rev4-4.h5", "r")
#data_file = h5py.File("extra_fine_discrete.h5", "r")


key = "foo"
TE_vals = data_file[key]["TE"]
surrogate_TE_vals = data_file[key]["surrogate_TE"]
num_events = data_file[key]["num_target_events"]
dt = data_file[key]["dt"].value
print(TE_vals.shape)

fig, axs = plt.subplots(nrows = TE_vals.shape[2], figsize = (8, 20))
plt.tight_layout()
for i in range(TE_vals.shape[2]):
    means = []
    stds = []
    surrogate_means = []
    surrogate_stds = []
    for j in range(TE_vals.shape[1]):
        cleaned = TE_vals[TE_vals[:, j, i] != -100, j, i]
        surrogate_cleaned = surrogate_TE_vals[TE_vals[:, j, i] != -100, j, i]
        #if i == 5:
        #    print(cleaned)
        means.append(np.mean(cleaned))
        stds.append(np.std(cleaned))
        surrogate_means.append(np.mean(surrogate_cleaned))
        surrogate_stds.append(np.std(surrogate_cleaned))

    means = np.array(means)
    stds = np.array(stds)
    surrogate_means = np.array(surrogate_means)
    surrogate_stds = np.array(surrogate_stds)



    sns.lineplot(x = num_events, y = means - surrogate_means, palette = "Set3", linewidth = 4, ax = axs[i])
    axs[i].fill_between(num_events, means - surrogate_means - stds, means - surrogate_means + stds, alpha = 0.5)
    #sns.lineplot(x = num_events, y = surrogate_means, palette = "Set3", linewidth = 4, ax = axs[i])
    axs[i].hlines(0.5076, 0, num_events[-1], color = "black", linewidth = 3)
    axs[i].hlines(0.0, 0.0, num_events[-1], color = "black", linewidth = 2, linestyle='--')

    axs[i].set_xscale("log")
    axs[i].set_ylim([-0.1, 2.5])
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
axs[5].set_xlabel("")
axs[6].set_xlabel("")
axs[7].set_xlabel("Number of Target Events")
plt.subplots_adjust(hspace = 0.4)

#plt.show()
plt.savefig("4c-rev4-4.pdf", bbox_inches='tight', format = "pdf")



plt.clf()
plt.rc('xtick', labelsize=18)
plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=20)
key = "foo"
histos = np.array(data_file[key]["histogram_freqs"])
fig, axs = plt.subplots(nrows = histos.shape[2], figsize = (8, 15))
plt.tight_layout()
print(histos.shape)
for i in range(histos.shape[2]):

    hist = histos[:, 2, i]
    print(hist)
    hist = np.array([val / sum(hist) for val in hist])
    #hist = np.expand_dims(hist, axis = 1)
    print(hist)
    print(list(range(HISTOGRAM_OF_FREQS_UPPER)))
    axs[i].bar(list(range(HISTOGRAM_OF_FREQS_UPPER)), hist)

    axs[i].set_title("$\Delta t$ = " + str(dt[i]))
    axs[i].set_ylim([0, 1])
    axs[i].set_xticks([0, 1, 3, 6, 9])
    axs[i].set_xticklabels(['0', '1', '3', '6', '>=9'])
    #axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #axs[i].xaxis.set_major_formatter(LogFormatterSciNotation('%.2f'))


axs[3].set_ylabel("Relative Frequency")
axs[0].set_xlabel("")
axs[1].set_xlabel("")
axs[2].set_xlabel("")
#axs[3].set_xlabel("Number of Target Events")
axs[3].set_xlabel("")
axs[4].set_xlabel("")
axs[5].set_xlabel("")
axs[6].set_xlabel("")
axs[7].set_xlabel("Number of 1's")
plt.subplots_adjust(hspace = 0.4)
#plt.show()
plt.savefig("4c-rev4-histo.pdf", bbox_inches='tight', format = "pdf")
