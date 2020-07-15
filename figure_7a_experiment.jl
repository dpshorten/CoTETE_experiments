using CSV: read
using HDF5: h5open, g_create
using Distances: Cityblock, Chebyshev


using CoTETE

d_y = 1
d_x = 1
d_c = 1
K = 10

#REPEATS = 10
REPEATS = 4
START_OFFSET = 10000
#TARGET_TRAIN_LENGTH = Int(5e4)
TARGET_TRAIN_LENGTH = Int(5e3)
NUM_SAMPLES_RATIO = 1.0
SURROGATE_UPSAMPLE_RATIO = 1.0
K_PERM = 10

NUM_SURROGATES = 20
NUM_SURROGATES = 100
MOTHER_NOISE_STD = 5e-2
DAUGHTER_NOISE_STDS = [7.5e-2, 5e-2]
MOTHER_T = 1
DAUGHTER_GAP_1 = 0.25
DAUGHTER_GAP_2 = 0.5

h5open(string("figure_7a.h5"), "w") do file

        for daughter_noise in DAUGHTER_NOISE_STDS
                # Watch out, the definition of a "positive" is different to paper
                for is_dependent in [false, true]
                        for i = 1:REPEATS
                                println()
                                println(daughter_noise, " ", is_dependent, " ", i)
                                TE = 0
                                surrogate_vals = zeros(NUM_SURROGATES)

                                generation_length = START_OFFSET + 3 * TARGET_TRAIN_LENGTH
                                mother_intervals =
                                        MOTHER_NOISE_STD .* randn(generation_length) .+ MOTHER_T
                                clamp!(mother_intervals, 0, 1e6)
                                mother_events = cumsum(mother_intervals)
                                daughter_events_1 = (
                                        mother_events +
                                        daughter_noise .* randn(generation_length) .+
                                        DAUGHTER_GAP_1
                                )
                                daughter_events_2 = (
                                        mother_events +
                                        daughter_noise .* randn(generation_length) .+
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

                                parameters = CoTETE.CoTETEParameters(
                                        l_x = d_x,
                                        l_y = d_y,
                                        l_z = d_c,
                                        auto_find_start_and_num_events = false,
                                        start_event = START_OFFSET,
                                        num_target_events = TARGET_TRAIN_LENGTH,
                                        num_samples_ratio = NUM_SAMPLES_RATIO,
                                        k_global = K,
                                        num_surrogates = NUM_SURROGATES,
                                )

                                TE, p, surrogates = CoTETE.estimate_TE_and_p_value_from_event_times(
                                        parameters,
                                        target_events,
                                        source_events,
                                        conditioning_events = conditioning_events,
                                        return_surrogate_TE_values = true,
                                )


                                sort!(surrogate_vals)
                                println("TE ", TE)
                                println(
                                        "surrogate ",
                                        surrogate_vals[1],
                                        surrogate_vals[90],
                                        surrogate_vals[end],
                                )
                                println(
                                        "p ",
                                        1 - (
                                                (searchsortedfirst(surrogate_vals, TE) - 1) /
                                                length(surrogate_vals)
                                        ),
                                )

                                g = g_create(file, string(daughter_noise, is_dependent, i))
                                g["TE"] = TE
                                g["TE_surrogate"] = surrogate_vals
                                g["num_events"] = TARGET_TRAIN_LENGTH
                                g["is_dependent"] = Int(is_dependent)
                                g["noise"] = daughter_noise
                                g["p"] =
                                        (searchsortedfirst(surrogate_vals, TE) - 1) /
                                        length(surrogate_vals)
                        end
                end
        end
end
