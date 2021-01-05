include("discretisation_testing.jl")


#TARGET_TRAIN_LENGTHS = [100, 1000, Int(1e4), Int(1e5), Int(1e6)]
TARGET_TRAIN_LENGTHS = [100, 1000, Int(1e4)]
REPETITIONS_PER_LENGTH = [1000, 100, 20, 20, 20]
DT_VALS = [1.0, 0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.005]
HISTOGRAM_OF_FREQS_UPPER = 10
OFFSET = 200

target_events = read(string("train_x_1.dat"))
source_events = read(string("train_y_1.dat"))

convert(Matrix, target_events)
target_events = target_events[:, 1]
convert(Matrix, source_events)
source_events = source_events[:, 1]

results_TE = []

h5open("figure_4c_rev4-4.h5", "w") do file
    TE_vals =
        -100 *
        ones((length(DT_VALS), length(TARGET_TRAIN_LENGTHS), maximum(REPETITIONS_PER_LENGTH)))
    surrogate_TE_vals =
        -100 *
        ones((length(DT_VALS), length(TARGET_TRAIN_LENGTHS), maximum(REPETITIONS_PER_LENGTH)))
    histogram_of_freqs =
        zeros((length(DT_VALS), length(TARGET_TRAIN_LENGTHS), HISTOGRAM_OF_FREQS_UPPER))
    for l = 1:length(DT_VALS)
        println("dt = ", DT_VALS[l])
        for i = 1:length(TARGET_TRAIN_LENGTHS)
            println("length = ", TARGET_TRAIN_LENGTHS[i])
            for j = 1:REPETITIONS_PER_LENGTH[i]
                target_start_event =
                    Int(round(OFFSET + ((j - 1) * (0.5 * TARGET_TRAIN_LENGTHS[i] + 20))))
                target_end_event = target_start_event + TARGET_TRAIN_LENGTHS[i]

                source_start_event = 1
                while source_events[source_start_event] < target_events[target_start_event]
                    source_start_event += 1
                end
                source_end_event = source_start_event
                while source_events[source_end_event] < target_events[target_end_event]
                    source_end_event += 1
                end
                source_end_event -= 1

                TE, joint_histogram_freqs, joint_histogram_keys = estimate_TE_discrete(
                    target_events[target_start_event:target_end_event],
                    source_events[source_start_event:source_end_event],
                    DT_VALS[l],
                    Int(round(1 / DT_VALS[l])),
                    Int(round(1 / DT_VALS[l])),
                    0,
                )

                TE_vals[l, i, j] = TE
                if j == 1
                    raw_freqs = [freq[2] for freq in joint_histogram_freqs]
                    for freq in raw_freqs
                        if freq >= HISTOGRAM_OF_FREQS_UPPER
                            histogram_of_freqs[l, i, HISTOGRAM_OF_FREQS_UPPER] += 1
                        else
                            histogram_of_freqs[l, i, freq+1] += 1
                        end
                    end
                end

                surrogate_source_events = source_events .- 20 * (1 + rand())
                clamp!(surrogate_source_events, 0, 1e6)
                surrogate_TE_vals[l, i, j] = estimate_TE_discrete(
                    target_events[target_start_event:target_end_event],
                    surrogate_source_events[source_start_event:source_end_event],
                    DT_VALS[l],
                    Int(round(1 / DT_VALS[l])),
                    Int(round(1 / DT_VALS[l])),
                    0,
                )[1]
            end
        end
    end

    for l = 1:length(DT_VALS)
        println(histogram_of_freqs[l, 3, :])
    end

    g = g_create(file, "foo")
    g["TE"] = TE_vals
    g["surrogate_TE"] = surrogate_TE_vals
    g["num_target_events"] = TARGET_TRAIN_LENGTHS
    g["dt"] = DT_VALS
    g["histogram_freqs"] = histogram_of_freqs
end
