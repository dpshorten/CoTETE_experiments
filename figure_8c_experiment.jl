using CSV: read
using HDF5: h5open, g_create
using Distances: Cityblock, Chebyshev, Euclidean
using Combinatorics: permutations
using Random: rand, randn
using StatsBase: sample

using CoTETE
#include("GLM_generative.jl")

l_x = 1
l_z = [1, 1, 1, 1]
l_y = 1

K = 10

START_OFFSET = 200
TARGET_TRAIN_LENGTH = Int(1e3)
#TARGET_TRAIN_LENGTH = Int(1e4)
METRIC = Cityblock()
NUM_SAMPLES_RATIO = 1.0
SURROGATE_UPSAMPLE_RATIO = 1.0
K_PERM = 5

#NUM_SURROGATES = 100
NUM_SURROGATES = 20

FOLDER = "stg_spike_files/"

h5open("figure_8c.h5", "w") do file
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
                conditioning_events =
                    conditioning_events + 1e-6 .* randn(size(conditioning_events)[1])
                sort!(conditioning_events)
                #conditioning_events = conditioning_events[1000:min(3 * TARGET_TRAIN_LENGTH, length(conditioning_events))]

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

                sort!(surrogates)
                println(
                    "surrogate ",
                    surrogates[1],
                    " ",
                    surrogates[Int(round(0.9 * NUM_SURROGATES))],
                    " ",
                    surrogates[Int(round(0.95 * NUM_SURROGATES))],
                    " ",
                    surrogates[end],
                )
                println()

                g = g_create(file, string(j, "_link_", permutation[3], permutation[1], folder))
                g["TE"] = TE
                g["folder"] = folder
                surrogates = Array{Float32}(surrogates)
                g["run"] = j
                g["surrogates"] = surrogates
                g["num_target_events"] = TARGET_TRAIN_LENGTH
                g["source"] = permutation[3]
                g["target"] = permutation[1]
    end
end
