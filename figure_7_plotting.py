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

plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=14)
plt.rc('figure', titlesize=18)

rc('text.latex', preamble=r'\usepackage{cmbright} \usepackage{amssymb}')


#data_file = h5py.File("figure_7a.h5", "r")
data_file = h5py.File("figure_7b.h5", "r")

P_CUTOFF = 0.95

num_events_vals = []
noise_vals = []
for key in data_file.keys():
    num_events = data_file[key]["num_events"].value
    noise = data_file[key]["noise"].value
    if num_events not in num_events_vals:
        num_events_vals.append(num_events)
    if noise not in noise_vals:
        noise_vals.append(noise)

num_events_vals.sort()
noise_vals.sort()
print(num_events_vals)
print(noise_vals)

p_vals = np.zeros((len(noise_vals), len(num_events_vals), 2, 10))

for i in range(len(noise_vals)):
    for j in range(len(num_events_vals)):
        for is_dependent in [True, False]:
            a = 0
            for key in data_file.keys():
                if (num_events_vals[j] == data_file[key]["num_events"].value and
                    noise_vals[i] == data_file[key]["noise"].value and
                    is_dependent == data_file[key]["is_dependent"].value):
                    p = data_file[key]["p"].value
                    p_vals[i, j, int(is_dependent), a] = 1 - p
                    a += 1


p_vals = np.reshape(p_vals, (len(noise_vals) * len(num_events_vals) *2, 10))
print(p_vals)
plt.ylabel("p value")
plt.ylim([-0.05, 1.0])

sns.boxplot(data = np.transpose(p_vals[:, :]), palette = "Set3",
             linewidth = 2, width = 0.5, fliersize = 10)
#plt.scatter(0, 0.93, s=1000, c='red', marker='$×$')
plt.scatter(0, 0.93, s=1000, c='green', marker='$✓$')
plt.scatter(1, 0.93, s=1000, c='green', marker='$✓$')
#plt.scatter(2, 0.93, s=1000, c='red', marker='$×$')
plt.scatter(2, 0.93, s=1000, c='green', marker='$✓$')
plt.scatter(3, 0.93, s=1000, c='green', marker='$✓$')
NAMES = ["$\\sigma_D$ = 7.5e-2\nzero flow", "$\\sigma_D$ = 7.5e-2\nnon-zero flow", "$\\sigma_D$ = 5e-2\nzero flow",
         "$\\sigma_D$ = 5e-2\nnon-zero flow"]
plt.xticks([0, 1, 2, 3], NAMES)
plt.show()
