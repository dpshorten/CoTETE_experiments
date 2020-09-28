import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plot_format

plot_format.set_format()

x = np.linspace(0, 1.0, 1000)
rate = 0.5 + 5 * np.exp(-50 * (x - 0.5)**2)

fig, ax = plt.subplots(figsize = (6, 8))
sns.lineplot(x = x, y = rate, linewidth = 4)
ax.set_facecolor('white')
ax.set_ylabel("$\lambda_{x|y}$")
ax.set_xlabel("$t_y^1$")
#plt.show()

plt.savefig("rate_func_plot.pdf", bbox_inches='tight', format = "pdf")
