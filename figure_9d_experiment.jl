include("discretisation_testing.jl")



REPEATS = 10

NUM_SURROGATES = 100
#NUM_SURROGATES = 20
START_OFFSET = 100
TARGET_TRAIN_LENGTH = Int(5e4)
#TARGET_TRAIN_LENGTH = Int(1e4)

DT = 0.05
#MAX_LAG = 20
MAX_LAG = 5
d_x = 7
d_y = 7
d_c = 7

h5open(string("figure_8d.h5"), "w") do file
    folder = "stg_spike_files/"
    target_start_event = START_OFFSET
    target_end_event = START_OFFSET + TARGET_TRAIN_LENGTH
    for j = 1:10
        println("*** ", j, " ***")
        println(folder)
        println()
        for permutation in collect(permutations(["abpd", "lp", "py"]))
            #for permutation in [["abpd", "py", "lp"], ["py", "abpd", "lp"]]
            #for permutation in [["abpd", "lp", "py"], ["lp", "abpd", "py"]]
            #for permutation in [["abpd", "lp", "py"], ["lp", "abpd", "py"], ["abpd", "py", "lp"], ["py", "abpd", "lp"]]

            target_events = read(string(folder, permutation[1], "_", j, ".dat"))
            conditioning_events = read(string(folder, permutation[2], "_", j, ".dat"))
            source_events = read(string(folder, permutation[3], "_", j, ".dat"))

            println(permutation[3], " ", permutation[1])

            convert(Matrix, target_events)
            target_events = target_events[:, 1]
            target_events = target_events + 1e-6 .* randn(size(target_events)[1])
            sort!(target_events)
            #target_events = target_events[1000:min(3 * TARGET_TRAIN_LENGTH, length(target_events))]
            convert(Matrix, source_events)
            source_events = source_events[:, 1]
            source_events = source_events + 1e-6 .* randn(size(source_events)[1])
            sort!(source_events)
            #source_events = source_events[1000:min(3 * TARGET_TRAIN_LENGTH, length(source_events))]
            convert(Matrix, conditioning_events)
            conditioning_events = conditioning_events[:, 1]
            conditioning_events = conditioning_events + 1e-6 .* randn(size(conditioning_events)[1])
            sort!(conditioning_events)

            TE = find_lags_and_calc_TE(
                target_events[target_start_event:target_end_event],
                source_events,
                conditioning_events,
                DT,
                d_x,
                d_y,
                d_c,
                MAX_LAG,
                MAX_LAG,
            )

            surrogate_vals = zeros(NUM_SURROGATES)

            Threads.@threads for j = 1:NUM_SURROGATES
                #source_events_surrogate = source_events .- 200 * (1 + rand())
                #clamp!(source_events_surrogate, 0, 1e6)

                TE_surrogate = find_lags_and_calc_TE(
                    target_events[target_start_event:target_end_event],
                    #source_events_surrogate,
                    source_events,
                    conditioning_events,
                    DT,
                    d_x,
                    d_y,
                    d_c,
                    MAX_LAG,
                    MAX_LAG,
                    permutation_surrogate = true,
                )

                surrogate_vals[j] = TE_surrogate
            end

            sort!(surrogate_vals)
            println("TE ", TE)
            println("surrogate ", surrogate_vals[1], " ", surrogate_vals[18], " ",
             surrogate_vals[end])
            println("p ", (searchsortedfirst(surrogate_vals, TE) - 1) / length(surrogate_vals))
            println()

            g = g_create(file, string(j, "_link_", permutation[3], permutation[1], folder))
            g["TE"] = TE
            g["folder"] = folder
            surrogates = Array{Float32}(surrogate_vals)
            g["run"] = j
            g["surrogates"] = surrogates
            g["num_target_events"] = TARGET_TRAIN_LENGTH
            g["source"] = permutation[3]
            g["target"] = permutation[1]

        end
    end
end
