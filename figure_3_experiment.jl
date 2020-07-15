include("discretisation_testing.jl")

HISTORY_LENGTHS = [2, 5]
TARGET_TRAIN_LENGTHS = [100, 1000, Int(1e4), Int(1e5)]
#TARGET_TRAIN_LENGTHS = [100, 1000, Int(1e4)]
REPETITIONS_PER_LENGTH = [1000, 100, 20, 20]
DT_VALS = [1.0, 0.5, 0.2, 0.1]


h5open(string("figure_3", ".h5"), "w") do file
    for m = 1:length(HISTORY_LENGTHS)
        TE_vals =
            -100 *
            ones((length(DT_VALS), length(TARGET_TRAIN_LENGTHS), maximum(REPETITIONS_PER_LENGTH)))
        for l = 1:length(DT_VALS)
            println(HISTORY_LENGTHS[m], " ", DT_VALS[l])
            for i = 1:length(TARGET_TRAIN_LENGTHS)
                Threads.@threads for j = 1:REPETITIONS_PER_LENGTH[i]
                    target_events = Float32(TARGET_TRAIN_LENGTHS[i]) * rand(TARGET_TRAIN_LENGTHS[i])
                    source_events = Float32(TARGET_TRAIN_LENGTHS[i]) * rand(TARGET_TRAIN_LENGTHS[i])
                    sort!(target_events)
                    sort!(source_events)
                    TE = estimate_TE_discrete(
                        target_events,
                        source_events,
                        DT_VALS[l],
                        HISTORY_LENGTHS[m],
                        HISTORY_LENGTHS[m],
                        0,
                    )
                    TE_vals[l, i, j] = TE
                end
            end
        end
        g = g_create(file, string(HISTORY_LENGTHS[m]))
        g["HL"] = HISTORY_LENGTHS[m]
        g["TE"] = TE_vals
        g["num_target_events"] = TARGET_TRAIN_LENGTHS
        g["dt"] = DT_VALS
    end
end
