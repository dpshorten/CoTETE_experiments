using CSV: File, read
using DataFrames
using HDF5: h5open, g_create
using Distances: Cityblock, Chebyshev, Euclidean
using Random: rand, randn
using StatsBase: sample

using CoTETE

l_x = 1
l_y = 1

K = 5

SIM_DT = 1e-4

START_OFFSET = 100
#TARGET_TRAIN_LENGTHS = [Int(1e2), Int(5e2), Int(1e3), Int(2e3)]
TARGET_TRAIN_LENGTHS = [Int(1e2), Int(5e2), Int(1e3), Int(2e3), Int(5e3), Int(1e4)]
#TARGET_TRAIN_LENGTH = Int(1e4)
METRIC = Cityblock()
NUM_SAMPLES_RATIO = 1.0
SURROGATE_UPSAMPLE_RATIO = 10.0
K_PERM = 10

NET_SIZES = [0, 1, 2]
CONDITIONING_SIZE = [6, 12, 18]
#EXTRA_TYPES = ["inh"]
EXTRA_TYPES = ["exc", "inh", "fake"]
#EXTRA_TYPES = ["fake"]
#EXTRA_TYPES = ["fake_corr"]

NUM_SURROGATES = 100

INPUT_FOLDER = "outputs_rev_exp_unison/"

h5open(string("uncorrelated_pop_unison_pairwise_auto/run_", ARGS[1], ".h5"), "w") do file
    for net_size in NET_SIZES
        for extra_type in EXTRA_TYPES
            for target_length in TARGET_TRAIN_LENGTHS

                l_z = ones(CONDITIONING_SIZE[net_size + 1])

                prefix = string(INPUT_FOLDER, "type_", extra_type, "_size_", net_size, "_net_", ARGS[1])

                target_events = DataFrame!(File((string(prefix, "_x_", ".dat"))))
                #target_events = read(string(prefix, "_x_", ".dat"))
                source_events = DataFrame!(File((string(prefix, "_y_", ".dat"))))
                array_of_conditioning_events = []
                for i = 1:CONDITIONING_SIZE[net_size + 1]
                    temp = DataFrame!(File((string(prefix, "_z__n_", i, ".dat"))))
                    push!(array_of_conditioning_events, temp)
                end

                convert(Matrix, target_events)
                target_events = target_events[:, 1]
                target_events += SIM_DT .* rand(size(target_events, 1)) .- 0.5 * SIM_DT
                convert(Matrix, source_events)
                source_events = source_events[:, 1]
                source_events += SIM_DT .* rand(size(source_events, 1)) .- 0.5 * SIM_DT

                new_cond = Array{Float32,1}[]
                for events in array_of_conditioning_events
                    convert(Matrix, events)
                    events = events[:, 1]
                    convert(Array{Float32,1}, events)
                    events += SIM_DT .* rand(size(events, 1)) .- 0.5 * SIM_DT
                    push!(new_cond, events)
                end

                parameters = CoTETE.CoTETEParameters(
                    l_x = l_x,
                    l_y = l_y,
                    #l_z = l_z,
                    auto_find_start_and_num_events = false,
                    start_event = START_OFFSET,
                    num_target_events = target_length,
                    num_samples_ratio = NUM_SAMPLES_RATIO,
                    k_global = K,
                    num_surrogates = NUM_SURROGATES,
                    add_dummy_exclusion_windows = false,
                    surrogate_num_samples_ratio = SURROGATE_UPSAMPLE_RATIO,
                    k_perm = K_PERM,
                    kraskov_noise_level = 1e-6,
                )

                TE, p, surrogates = CoTETE.estimate_TE_and_p_value_from_event_times(
                    parameters,
                    target_events,
                    source_events,
                    #conditioning_events = new_cond,
                    return_surrogate_TE_values = true,
                )

                println(extra_type, " ", net_size, " ", target_length)

                println(p)

                println(TE)
                sort!(surrogates)
                println()
                #println(surrogates)

                g = g_create(file, string(net_size, extra_type, target_length))
                g["TE"] = TE
                surrogates = Array{Float32}(surrogates)
                g["p"] = p
                g["net_size"] = net_size
                g["extra_type"] = extra_type
                g["target_length"] = target_length
            end
        end
    end
end
