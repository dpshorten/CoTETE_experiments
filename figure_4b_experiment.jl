using CSV: read
using HDF5: h5open, g_create
using Distances: Cityblock, Chebyshev, Euclidean

using CoTETE

FIGURE_TYPES = ["main", "high_d_y", "extra_reps"]
FIGURE_TYPE_INDEX = 1

D_Y = 1
if FIGURE_TYPE_INDEX == 2
    D_Y = 3
end
K = 4
D_X_VALS = [1, 2, 3]


START_OFFSET = 5000
TARGET_TRAIN_LENGTHS = [Int(1e2), Int(1e3), Int(1e4), Int(1e5), Int(1e6)]
REPETITIONS_PER_LENGTH = [1000, 100, 20, 20, 20]
if FIGURE_TYPE_INDEX == 3
    REPETITIONS_PER_LENGTH = [1000, 100, 100, 100, 100]
end

h5open(string("figure_4b_", FIGURE_TYPES[FIGURE_TYPE_INDEX],".h5"), "w") do file

    target_events = read("train_x_1.dat")
    source_events = read("train_y_1.dat")

    convert(Matrix, target_events)
    target_events = target_events[:, 1]
    convert(Matrix, source_events)
    source_events = source_events[:, 1]

    TE_vals =
        -100 *
        ones((length(D_X_VALS), length(TARGET_TRAIN_LENGTHS), maximum(REPETITIONS_PER_LENGTH)))
    for d_x in D_X_VALS
        for i = 1:length(TARGET_TRAIN_LENGTHS)
            println(d_x, " ", i)
            Threads.@threads for j = 1:REPETITIONS_PER_LENGTH[i]
                parameters = CoTETE.CoTETEParameters(
                    l_x = d_x,
                    l_y = D_Y,
                    auto_find_start_and_num_events = false,
                    start_event = START_OFFSET + (j * TARGET_TRAIN_LENGTHS[i]),
                    num_target_events = TARGET_TRAIN_LENGTHS[i],
                    num_samples_ratio = 1.0,
                    k_global = K,
                )
                TE = CoTETE.estimate_TE_from_event_times(parameters, target_events, source_events)
                TE_vals[d_x, i, j] = TE
            end
        end
    end

    g = g_create(file, string("bar"))
    g["TE"] = TE_vals
    g["d_x"] = D_X_VALS
    g["num_target_events"] = TARGET_TRAIN_LENGTHS

end
