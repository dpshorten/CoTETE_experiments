using CSV: read
using HDF5: h5open, g_create
using Distances: Cityblock, Chebyshev, Euclidean


using CoTETE

d_y = 1
K = 4

d_x_VALS = [1, 2, 3]


START_OFFSET = 5000
TARGET_TRAIN_LENGTHS = [Int(1e2), Int(1e3), Int(1e4), Int(1e5)]
REPETITIONS_PER_LENGTH = [1000, 100, 20, 20]

h5open("run_outputs/figure_4b.h5", "w") do file

    target_events = read("canonical_example_dats/train_x_1.dat")
    source_events = read("canonical_example_dats/train_y_1.dat")

    convert(Matrix, target_events)
    target_events = target_events[:, 1]
    convert(Matrix, source_events)
    source_events = source_events[:, 1]

    TE_vals = -100 * ones((length(d_x_VALS), length(TARGET_TRAIN_LENGTHS), maximum(REPETITIONS_PER_LENGTH)))
    for d_x in d_x_VALS
        for i = 1:length(TARGET_TRAIN_LENGTHS)
            Threads.@threads for j = 1:REPETITIONS_PER_LENGTH[i]
                TE = CoTETE.calculate_TE_from_event_times(
                    target_events,
                    source_events,
                    d_x,
                    d_y,
                    auto_find_start_and_num_events = false,
                    num_target_events = TARGET_TRAIN_LENGTHS[i],
                    num_samples_ratio = 1.0,
                    k_global = K,
                    start_event = START_OFFSET + (j * TARGET_TRAIN_LENGTHS[i]),
                    metric = Cityblock(),
                )

                TE_vals[d_x, i, j] = TE
            end
        end
    end

    g = g_create(file, string("bar"))
    g["TE"] = TE_vals
    g["d_x"] = d_x_VALS
    g["num_target_events"] = TARGET_TRAIN_LENGTHS

end
