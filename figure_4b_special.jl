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
DT_VALS = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]


START_OFFSET = 5000
TARGET_TRAIN_LENGTHS = [Int(1e2), Int(1e3), Int(1e4), Int(1e5)]
REPETITIONS_PER_LENGTH = [1000, 100, 20, 20, 20]
if FIGURE_TYPE_INDEX == 3
    REPETITIONS_PER_LENGTH = [1000, 100, 100, 100, 100]
end

h5open(string("figure_4_special", FIGURE_TYPES[FIGURE_TYPE_INDEX],".h5"), "w") do file

    TE_vals =
        -100 *
        ones((length(DT_VALS), length(TARGET_TRAIN_LENGTHS), maximum(REPETITIONS_PER_LENGTH)))
    for l = 1:length(DT_VALS)
        dt = DT_VALS[l]
        target_events = read("train_x_1.dat")
        source_events = read("train_y_1.dat")

        convert(Matrix, target_events)
        target_events = [(dt * round(time/dt)) for time in target_events[:, 1]]
        target_events += (rand(size(target_events, 1)) .* dt) .- (dt/2)
        sort!(target_events)
        convert(Matrix, source_events)
        source_events = [(dt * round(time/dt)) for time in source_events[:, 1]]
        source_events += (rand(size(source_events, 1)) .* dt) .- (dt/2)
        sort!(source_events)

        for i = 1:length(TARGET_TRAIN_LENGTHS)
            println(dt, " ", i)
            Threads.@threads for j = 1:REPETITIONS_PER_LENGTH[i]
                parameters = CoTETE.CoTETEParameters(
                    l_x = 2,
                    l_y = D_Y,
                    auto_find_start_and_num_events = false,
                    start_event = START_OFFSET + (j * TARGET_TRAIN_LENGTHS[i]),
                    num_target_events = TARGET_TRAIN_LENGTHS[i],
                    num_samples_ratio = 1.0,
                    k_global = K,
                )
                TE = CoTETE.estimate_TE_from_event_times(parameters, target_events, source_events)
                TE_vals[l, i, j] = TE
            end
        end
    end

    g = g_create(file, string("bar"))
    g["TE"] = TE_vals
    g["dt"] = DT_VALS
    g["num_target_events"] = TARGET_TRAIN_LENGTHS

end
