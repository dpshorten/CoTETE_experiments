import matplotlib.pyplot as plt
import numpy as np
import h5py
import seaborn as sns
import plot_format

plot_format.set_format()

data_file = h5py.File("figure_6t.h5", "r")

key = "foo"
TE_vals = data_file[key]["TE"].value
TE_surrogates = data_file[key]["TE_surrogate"].value
TE_shift_surrogates = data_file[key]["TE_shift_surrogate"].value
shifts = data_file[key]["shifts"].value

mean = np.mean(TE_vals, axis = 1)
std = np.std(TE_vals, axis = 1)
mean_surrogates = np.mean(TE_surrogates, axis = 1)
std_surrogates = np.std(TE_surrogates, axis = 1)
mean_shift_surrogates = np.mean(TE_shift_surrogates, axis = 1)
std_shift_surrogates = np.std(TE_shift_surrogates, axis = 1)

plt.clf()
plt.figure(figsize=(12,6))
sns.lineplot(x = shifts, y = mean_surrogates, palette = "Set3", linewidth = 4, label = "Local\nPermutation\nSurrogate")
plt.fill_between(shifts, mean_surrogates - std_surrogates, mean_surrogates + std_surrogates, alpha = 0.5)
sns.lineplot(x = shifts, y = mean_shift_surrogates, palette = "Set3", linewidth = 4, label = "Shifted\nSurrogate")
plt.fill_between(shifts, mean_shift_surrogates - std_shift_surrogates, mean_shift_surrogates + std_shift_surrogates, alpha = 0.5)
sns.lineplot(x = shifts, y = mean, palette = "Set3", linewidth = 4, label = "TE")
plt.fill_between(shifts, mean - std, mean + std, alpha = 0.5)

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.xlabel("shift")
plt.ylabel("TE(nats/second)")

plt.tight_layout()
#plt.show()

plt.savefig("shifts_t.pdf", bbox_inches='tight', format = 'pdf')
