include("discretisation_testing.jl")


REPEATS = 10

NUM_SURROGATES = 100
#NUM_SURROGATES = 20
START_OFFSET = 100
#TARGET_TRAIN_LENGTHS = [Int(5e4)]
TARGET_TRAIN_LENGTHS = [Int(2e3)]
MOTHER_NOISE_STD = 5e-2
DAUGHTER_NOISE_STDS = [7.5e-2, 5e-2]
MOTHER_T = 1
DAUGHTER_GAP_1 = 0.25
DAUGHTER_GAP_2 = 0.5

DT = 0.05
MAX_LAG = 10
d_x = 7
d_y = 7
d_c = 7

h5open(string("figure_7b.h5"), "w") do file
    for target_length in TARGET_TRAIN_LENGTHS
        for daughter_noise in DAUGHTER_NOISE_STDS
            # Watch out, the definition of a "positive" is different to paper
            for is_dependent in [false, true]
                for i = 1:REPEATS
                    println()
                    println(daughter_noise, " ", is_dependent, " ", target_length, " ", i)
                    TE = 0
                    surrogate_vals = zeros(NUM_SURROGATES)
                    generation_length = START_OFFSET + 3 * target_length
                    mother_intervals = MOTHER_NOISE_STD .* randn(generation_length) .+ MOTHER_T
                    clamp!(mother_intervals, 0, 1e6)
                    mother_events = cumsum(mother_intervals)
                    daughter_events_1 = (
                        mother_events + daughter_noise .* randn(generation_length) .+
                        DAUGHTER_GAP_1
                    )
                    daughter_events_2 = (
                        mother_events + daughter_noise .* randn(generation_length) .+
                        DAUGHTER_GAP_2
                    )
                    sort!(daughter_events_1)
                    sort!(daughter_events_2)

                    if is_dependent
                        source_events = mother_events
                        conditioning_events = daughter_events_1
                        target_events = daughter_events_2
                    else
                        source_events = daughter_events_1
                        conditioning_events = mother_events
                        target_events = daughter_events_2
                    end

                    target_start_event = START_OFFSET
                    target_end_event = START_OFFSET + target_length

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


                    Threads.@threads for j = 1:NUM_SURROGATES
                        source_events_surrogate = source_events .- 200 * (1 + rand())
                        clamp!(source_events_surrogate, 0, 1e6)

                        TE_surrogate = find_lags_and_calc_TE(
                            target_events[target_start_event:target_end_event],
                            source_events_surrogate,
                            conditioning_events,
                            DT,
                            d_x,
                            d_y,
                            d_c,
                            MAX_LAG,
                            MAX_LAG,
                        )

                        surrogate_vals[j] = TE_surrogate
                    end

                    sort!(surrogate_vals)
                    println("TE ", TE)
                    #println(
                    #    "surrogate ",
                    #    surrogate_vals[1],
                    #    surrogate_vals[90],
                    #    surrogate_vals[end],
                    #)
                    println(
                        "p ",
                        (searchsortedfirst(surrogate_vals, TE) - 1) / length(surrogate_vals),
                    )

                    g = g_create(file, string(daughter_noise, is_dependent, target_length, i))
                    g["TE"] = TE
                    g["TE_surrogate"] = surrogate_vals
                    g["num_events"] = target_length
                    g["is_dependent"] = Int(is_dependent)
                    g["noise"] = daughter_noise
                    g["p"] = (searchsortedfirst(surrogate_vals, TE) - 1) / length(surrogate_vals)

                end
            end
        end
    end
end
