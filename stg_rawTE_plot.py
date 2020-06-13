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

rc('text.latex', preamble=r'\usepackage{cmbright}')


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

FOLDERS = ["output_stg_full17/"]

NAMES = ["min"]

#data_file = h5py.File("run_outputs/foo_discrete_min_1.h5", "r")
data_file = h5py.File("run_outputs/stg_foo_bar_4_min.h5", "r")

for i in range(len(FOLDERS)):

   plt.clf()

   TE = np.zeros((len(links), num_runs))
   surrogates = np.zeros((len(links), num_runs, num_surrogates))

   for key in data_file.keys():
      if str(data_file[key]["folder"].value)[2:-1] == FOLDERS[i]:
         source = str(data_file[key]["source"].value)[2:-1]
         target = str(data_file[key]["target"].value)[2:-1]
         num_targets = data_file[key]["num_target_events"].value
         run = data_file[key]["run"].value

         TE[links.index(source + "-" + target), run - 1] = data_file[key]["TE"].value

   print(TE.shape)

   sns.boxplot(data = np.transpose(TE[:, :]), palette = "Set3", linewidth = 2, width = 0.5, fliersize = 4)
   plt.xticks([0, 1, 2, 3, 4, 5], LINKS)

   plt.xlabel("connection", fontsize = 14)
   plt.ylabel("TE Rate (nats/s)", fontsize = 14)

   plt.tight_layout()
   plt.savefig("figures/stg_raw_" + NAMES[i])
