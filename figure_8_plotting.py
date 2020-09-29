import matplotlib.pyplot as plt
import numpy as np
import h5py
import seaborn as sns
import plot_format

plot_format.set_format()



NUM_RUNS = 30
SIZES = [6, 12, 18]
TARGET_TRAIN_LENGTHS = [int(1e2), int(5e2), int(1e3), int(2e3), int(5e3), int(1e4)]

OUTPUT_FILES = [
"continuous_multi_correlated",
"continuous_multi_uncorrelated",
"continuous_pair_correlated",
"continuous_pair_uncorrelated",
"discrete_multi_correlated",
"discrete_multi_uncorrelated",
"discrete_pair_correlated",
"discrete_pair_uncorrelated",
]

INPUT_FILES = [
"correlated_pop_unison",
"uncorrelated_pop_unison",
"correlated_pop_unison_pairwise",
"uncorrelated_pop_unison_pairwise",
"correlated_pop_discrete_unison",
"uncorrelated_pop_discrete_unison",
"correlated_pop_discrete_unison_pairwise",
"uncorrelated_pop_discrete_unison_pairwise",
]

def make_heatmap(title, filename, p_vals):
    #plt.rc('xtick', labelsize=18)
    fig, axs = plt.subplots(figsize = (10, 7))
    sns.heatmap(p_vals, vmin = 0, vmax = 1)
    plt.title(title)
    plt.xticks(ticks = np.arange(len(TARGET_TRAIN_LENGTHS)) + 0.5, labels = TARGET_TRAIN_LENGTHS)
    plt.yticks(ticks = np.flip(np.arange(3) + 0.5), labels = SIZES)
    plt.xlabel("num target spikes")
    plt.ylabel("num conditioning processes")
    plt.savefig("figures/" + filename + ".pdf",
                bbox_inches='tight', format = 'pdf')
    plt.show()


for k in range(len(OUTPUT_FILES)):
    exc_p = np.zeros((NUM_RUNS, len(SIZES), len(TARGET_TRAIN_LENGTHS)))
    inh_p = np.zeros((NUM_RUNS, len(SIZES), len(TARGET_TRAIN_LENGTHS)))
    fake_p = np.zeros((NUM_RUNS, len(SIZES), len(TARGET_TRAIN_LENGTHS)))
    fake_corr_p = np.zeros((NUM_RUNS, len(SIZES), len(TARGET_TRAIN_LENGTHS)))

    for i in range(NUM_RUNS):

        data_file = h5py.File(INPUT_FILES[k] + "/run_" + str(i + 1) + ".h5", "r")

        for key in data_file.keys():
            p = data_file[key]["p"].value
            net_size_index = len(SIZES) - data_file[key]["net_size"].value - 1
            extra_type = str(data_file[key]["extra_type"].value)[2:-1]
            target_length = data_file[key]["target_length"].value
            target_length_index = TARGET_TRAIN_LENGTHS.index(target_length)

            if extra_type == "exc":
                exc_p[i, net_size_index, target_length_index] = p
            elif extra_type == "inh":
                inh_p[i, net_size_index, target_length_index] = p
            else:
                fake_p[i, net_size_index, target_length_index] = p



    #exc_p = np.mean(exc_p, axis = 0)
    exc_p = exc_p < 0.05
    inh_p = inh_p < 0.05
    fake_p = fake_p < 0.05
    fake_corr_p = fake_corr_p < 0.05
    exc_p = np.sum(exc_p, axis = 0)/NUM_RUNS
    inh_p = np.sum(inh_p, axis = 0)/NUM_RUNS
    fake_p = np.sum(fake_p, axis = 0)/NUM_RUNS
    fake_corr_p = np.sum(fake_corr_p, axis = 0)/NUM_RUNS




    make_heatmap("Excitatory true positive rate", OUTPUT_FILES[k] + "/cont_exc", exc_p)
    make_heatmap("Inhibitory true positive rate", OUTPUT_FILES[k] + "/cont_inh", inh_p)
    make_heatmap("False positive rate", OUTPUT_FILES[k] + "/cont_fake", fake_p)
