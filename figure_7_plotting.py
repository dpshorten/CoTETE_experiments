import matplotlib.pyplot as plt
import numpy as np
import h5py
import seaborn as sns
import plot_format

plot_format.set_format()

plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=18)


FIGURE_TYPES = ["a", "b"]
FIGURE_TYPE_INDEX = 1

if FIGURE_TYPE_INDEX == 1:
    data_file = h5py.File("figure_7a.h5", "r")
else:
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
#
plt.ylabel("p value")


sns.boxplot(data = np.transpose(p_vals[:, :]), palette = "colorblind",
             linewidth = 4, width = 0.5, fliersize = 10)
sns.stripplot(data = np.transpose(p_vals[:, :]), palette = "colorblind",
             linewidth = 3, size = 10)
plt.hlines(0.05, -0.5, 3.5, color = "black", linewidth = 2, linestyle='--')
plt.xlim([-0.5, 3.5])

if FIGURE_TYPE_INDEX == 1:
    plt.scatter(0, 0.93, s=1000, c='green', marker='$✓$')
    plt.scatter(2, 0.93, s=1000, c='green', marker='$✓$')
else:
    plt.scatter(0, 0.93, s=1000, c='red', marker='$×$')
    plt.scatter(2, 0.93, s=1000, c='red', marker='$×$')

plt.scatter(1, 0.93, s=1000, c='green', marker='$✓$')
plt.scatter(3, 0.93, s=1000, c='green', marker='$✓$')

plt.ylim([-0.1, 1.0])
#plt.yscale("symlog")

NAMES = ["$\\sigma_D \\! = \\! 7.5$e-2\n $D_1 \\rightarrow D_2 | M$",
         "$\\sigma_D \\! = \\! 7.5$e-2\n  $M \\rightarrow D_2 | D_1$",
         "$\\sigma_D \\! = \\! 5$e-2\n  $D_1 \\rightarrow D_2 | M$",
         "$\\sigma_D \\! = \\! 5$e-2\n  $M \\rightarrow D_2 | D_1$",]
plt.xticks([0, 1, 2, 3], NAMES)

if FIGURE_TYPE_INDEX == 1:
    save_file_name = "noisy_copy_inference_continuous.pdf"
else:
    save_file_name = "noisy_copy_inference_discrete.pdf"

plt.savefig(save_file_name,
            bbox_inches='tight', format = 'pdf')
