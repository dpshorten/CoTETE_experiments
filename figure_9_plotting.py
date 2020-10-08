import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import seaborn as sns
import plot_format

plot_format.set_format()

plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=18)


x = [int(2.5e3)]
num_runs = 9
num_surrogates = 100

links = [
   "abpd-lp",
   "abpd-py",
   "lp-abpd",
   "lp-py",
   "py-abpd",
   "py-lp",
]

LINKS = [
   "ABPD\nto\nLP",
   "ABPD\nto\nPY",
   "LP\nto\nABPD",
   "LP\nto\nPY",
   "PY\nto\nABPD",
   "PY\nto\nLP",
]

data_file = h5py.File("new_nines/figure_9c_3e4.h5", "r")

plt.clf()

TE = np.zeros((len(links), num_runs + 1))
surrogates = np.zeros((len(links), num_runs + 1, num_surrogates))

for key in data_file.keys():
   source = str(data_file[key]["source"].value)[2:-1]
   target = str(data_file[key]["target"].value)[2:-1]
   num_targets = data_file[key]["num_target_events"].value
   run = data_file[key]["run"].value

   TE[links.index(source + "-" + target), run - 1] = data_file[key]["TE"].value
   surrogates[links.index(source + "-" + target), run - 1, :] = data_file[key]["surrogates"].value

print(TE.shape)
print(surrogates.shape)

print(TE[4, 0])
print(surrogates[4, 0, :])

p_vals = np.zeros((len(links), num_runs))
for l in range(len(links)):
   for r in range(num_runs):
      if 1 in (surrogates[l, r, :] > TE[l, r]):
         p_vals[l, r] = 1-(np.argmax(surrogates[l, r, :] > TE[l, r])/num_surrogates)
      else:
         p_vals[l, r] = 0

print(p_vals)
p_vals = np.delete(p_vals, obj = 1, axis = 1)
print(p_vals)

fig, axs = plt.subplots(figsize = (6, 6))
#sns.boxplot(data = np.transpose(p_vals[:, :]), palette = "Set3", linewidth = 2, width = 0.5, fliersize = 4)
sns.boxplot(data = np.transpose(p_vals[:, :]), palette = "colorblind",
             linewidth = 4, width = 0.5, fliersize = 0)
sns.stripplot(data = np.transpose(p_vals[:, :]), palette = "colorblind",
             linewidth = 3, size = 10)
plt.hlines(0.05, -0.5, 5.5, color = "black", linewidth = 2, linestyle='--')
plt.xticks([0, 1, 2, 3, 4, 5], LINKS)

#plt.xlabel("connection")
plt.ylabel("p value")
plt.ylim([-0.1, 1.19])

#for i in [0, 1, 2, 3, 5]:
for i in range(6):
   plt.scatter(i, 1.1, s=1000, c='green', marker='$✓$')
#for i in [4]:
#   plt.scatter(i, 1.1, s=1000, c='red', marker='$×$')


plt.tight_layout()
#plt.show()

plt.savefig("stg_sig_full_3point5.pdf",
            bbox_inches='tight', format = 'pdf')
#plt.savefig("stg_sig_full")
