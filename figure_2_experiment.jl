using CSV: read
using HDF5: h5open, g_create
using Distances: Cityblock, Chebyshev, Euclidean

using CoTETE

d_y = 1
d_x = 1
K = [1, 5]

MU = [0.5, 1, 2, 5]

START_OFFSET = 5000
#TARGET_TRAIN_LENGTHS = [Int(1e2), Int(1e3), Int(1e4), Int(1e5)]
TARGET_TRAIN_LENGTHS = [Int(1e2), Int(1e3), Int(1e4)]
REPETITIONS_PER_LENGTH = [100, 20, 20, 20]

h5open(string("run_outputs/figure_2", ".h5"), "w") do file

    target_events = 1e7 * rand(Int(1e7))
    sort!(target_events)
    source_events = 1e7 * rand(Int(1e7))
    sort!(source_events)

    for k in K
        TE_vals = -100 * ones((length(MU), length(TARGET_TRAIN_LENGTHS), maximum(REPETITIONS_PER_LENGTH)))
        mu_ind = 0
        for mu in MU
            println("mu ", mu)
            mu_ind += 1
            for i = 1:length(TARGET_TRAIN_LENGTHS)
                Threads.@threads for j = 1:REPETITIONS_PER_LENGTH[i]

                    TE = CoTETE.calculate_TE_from_event_times(
                        target_events,
                        source_events,
                        d_x,
                        d_y,
                        auto_find_start_and_num_events = false,
                        num_target_events = TARGET_TRAIN_LENGTHS[i],
                        num_samples_ratio =  mu,
                        k_global = k,
                        start_event = START_OFFSET + (j * TARGET_TRAIN_LENGTHS[i]),
                        metric = Cityblock(),
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
