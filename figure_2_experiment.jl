using CSV: read
using HDF5: h5open, g_create
using Distances: Cityblock, Chebyshev, Euclidean

using CoTETE

FIGURE_TYPES = ["main", "high_d_y", "extra_reps"]
FIGURE_TYPE_INDEX = 1

L_Y = 1
L_X = 1

if FIGURE_TYPE_INDEX == 2
    L_Y = 3
    L_X = 3
end


K = [1, 5]

MU = [0.5, 1, 2, 5]

START_OFFSET = 5000
#TARGET_TRAIN_LENGTHS = [Int(1e2), Int(1e3), Int(1e4), Int(1e5)]
TARGET_TRAIN_LENGTHS = [Int(1e2), Int(1e3), Int(1e4), Int(1e5)]
REPETITIONS_PER_LENGTH = [1000, 100, 20, 20]
if FIGURE_TYPE_INDEX == 3
    REPETITIONS_PER_LENGTH = [1000, 100, 100, 100]
end

h5open(string("figure_2_", FIGURE_TYPES[FIGURE_TYPE_INDEX], ".h5"), "w") do file

    target_events = 1e7 * rand(Int(1e7))
    sort!(target_events)
    source_events = 1e7 * rand(Int(1e7))
    sort!(source_events)

    for k in K
        TE_vals =
            -100 * ones((length(MU), length(TARGET_TRAIN_LENGTHS), maximum(REPETITIONS_PER_LENGTH)))
        mu_ind = 0
        for mu in MU
            println("k ", k, "mu ", mu)
            mu_ind += 1
            for i = 1:length(TARGET_TRAIN_LENGTHS)
                Threads.@threads for j = 1:REPETITIONS_PER_LENGTH[i]
                    parameters = CoTETE.CoTETEParameters(
                        l_x = L_X,
                        l_y = L_Y,
                        auto_find_start_and_num_events = false,
                        start_event = START_OFFSET + (j * TARGET_TRAIN_LENGTHS[i]),
                        num_target_events = TARGET_TRAIN_LENGTHS[i],
                        num_samples_ratio = mu,
                        k_global = k,
                    )
                    TE = CoTETE.estimate_TE_from_event_times(
                        parameters,
                        target_events,
                        source_events,
                    )

                    TE_vals[mu_ind, i, j] = TE
                end
            end
        end

        g = g_create(file, string(k))
        g["k"] = k
        g["TE"] = TE_vals
        g["num_target_events"] = TARGET_TRAIN_LENGTHS
        g["mu"] = MU
    end
end
