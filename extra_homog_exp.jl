include("discretisation_testing.jl")


TARGET_TRAIN_LENGTHS = [100, 1000, Int(1e4), Int(1e5)]
#TARGET_TRAIN_LENGTHS = [100, 1000, Int(1e4)]
REPETITIONS_PER_LENGTH = [1000, 100, 20, 20]
DT_VALS = [1.0, 0.5, 0.2, 0.1]
HISTORY_LENGTHS = [1, 2, 5, 10]

h5open(string("extra_homog", ".h5"), "w") do file
    TE_vals =
    -100 *
    ones((length(DT_VALS), length(TARGET_TRAIN_LENGTHS), maximum(REPETITIONS_PER_LENGTH)))
    for l = 1:length(DT_VALS)
        println(HISTORY_LENGTHS[l], " ", DT_VALS[l])
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
                HISTORY_LENGTHS[l],
                HISTORY_LENGTHS[l],
                0,
                )
                TE_vals[l, i, j] = TE
            end
        end
    end
    g = g_create(file, "foo")
    g["HL"] = 1
    g["TE"] = TE_vals
    g["num_target_events"] = TARGET_TRAIN_LENGTHS
    g["dt"] = DT_VALS
end
