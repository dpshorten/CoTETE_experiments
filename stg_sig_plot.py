import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import sklearn.metrics
import seaborn as sns

#plt.style.use('seaborn-dark-palette')

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],
             'size' : 18})

plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=14)
plt.rc('figure', titlesize=18)

rc('text.latex', preamble=r'\usepackage{cmbright}')

#x = [int(1e3), int(2.5e3), int(5e3), int(1e4), int(2e4)]
x = [int(2.5e3)]
num_runs = 10
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
   "ABPD-LP",
   "ABPD-PY",
   "LP-ABPD",
   "LP-PY",
   "PY-ABPD",
   "PY-LP",
]

markers = [
   '1',
   'p',
   's',
   '*',
   '+',
   'x',
   ]

sizes = [
   300,
   150,
   150,
   150,
   300,
   200,
   ]

thicknesses = [
   3,
   1,
   1,
   1,
   3,
   3,
   ]

FOLDERS = ["output_stg_full17/"]

NAMES = ["min"]

#data_file = h5py.File("run_outputs/stg_foo_bar_4_min.h5", "r")
data_file = h5py.File("run_outputs/connectivity_discrete_full.h5", "r")

for i in range(len(FOLDERS)):

   plt.clf()

   TE = np.zeros((len(links), num_runs))
   surrogates = np.zeros((len(links), num_runs, num_surrogates))

   for key in data_file.keys():
      print(str(data_file[key]["folder"].value))
      if str(data_file[key]["folder"].value)[2:-1] == FOLDERS[i]:
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
   means_p_vals = np.mean(p_vals, axis = 1)
   stds_p_vals = np.std(p_vals, axis = 1)
   print()
   print(p_vals[4, :])

   sns.boxplot(data = np.transpose(p_vals[:, :]), palette = "Set3", linewidth = 2, width = 0.5, fliersize = 4)
   plt.xticks([0, 1, 2, 3, 4, 5], LINKS)

   plt.xlabel("connection", fontsize = 14)
   plt.ylabel("p value", fontsize = 14)
   plt.ylim([-0.1, 1.19])

   for i in [0, 1, 2, 5]:
   #for i in range(6):
      plt.scatter(i, 1.1, s=1000, c='green', marker='$✓$')
   for i in [3, 4]:
      plt.scatter(i, 1.1, s=1000, c='red', marker='$×$')


   plt.tight_layout()
   plt.show()
   #plt.savefig("figures/stg_sig_full")
