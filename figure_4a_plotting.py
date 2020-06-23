import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

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

x = np.linspace(0, 1.0, 1000)
rate = 0.5 + 5 * np.exp(-50 * (x - 0.5)**2)


fig, ax = plt.subplots(figsize = (6, 8))
ax.plot(x, rate, linewidth = 3)
ax.set_facecolor('white')
ax.set_ylabel("$\lambda_{x|y}$")
ax.set_xlabel("$t_y^1$")
plt.show()

plt.savefig("rate", bbox_inches='tight')

