import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plot_format

plot_format.set_format()


spike_times = np.genfromtxt('output_pyloric_noisy3/py_1.dat', delimiter = '\n')
print(spike_times[:10])

intervals = spike_times[1:] - spike_times[:-1]
print(intervals[:10])

sns.distplot(a = intervals, kde = False, bins = np.linspace(0, 1.5, num = 50),
color = "blue", hist_kws=dict(alpha=0.6))
plt.xlim([0, 1.5])
plt.xlabel("Interspike Interval (seconds)")
plt.ylabel("Frequency")

plt.savefig("ISI.pdf",
            bbox_inches='tight', format = 'pdf')
