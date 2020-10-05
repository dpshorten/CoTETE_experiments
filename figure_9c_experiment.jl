using CSV: read
using HDF5: h5open, g_create
using Distances: Cityblock, Chebyshev, Euclidean
using Combinatorics: permutations
using Random: rand, randn
using StatsBase: sample

using CoTETE
#include("GLM_generative.jl")

l_x = 1
l_z = [1]
l_y = 1

K = 5

START_OFFSET = 100
TARGET_TRAIN_LENGTH = Int(3e3)
#TARGET_TRAIN_LENGTH = Int(1e4)
METRIC = Cityblock()
NUM_SAMPLES_RATIO = 1.0
SURROGATE_UPSAMPLE_RATIO = 3.1
K_PERM = 5

NUM_SURROGATES = 100
#NUM_SURROGATES = 20

folder = "output_pyloric_noisy2/"

h5open("figure_8c.h5", "w") do file
    for j = 1:10
    for permutation in permutations(["abpd", "py", "lp"])
    #for permutation in [["abpd", "lp", "py"]]
        target_events = read(string(folder, permutation[1], "_", j, ".dat"))
        conditioning_events = read(string(folder, permutation[2], "_", j, ".dat"))
        source_events = read(string(folder, permutation[3], "_", j, ".dat"))

        println(permutation[3], " ", permutation[1])

        convert(Matrix, target_events)
        target_events = target_events[:, 1]
	target_events = target_events + 1e-4 .* (rand(size(target_events)[1]) .- 0.5)
        sort!(target_events)
        #target_events = target_events[1000:min(3 * TARGET_TRAIN_LENGTH, length(target_events))]
        convert(Matrix, source_events)
        source_events = source_events[:, 1]
        source_events = source_events + 1e-4 .* (rand(size(source_events)[1]) .- 0.5)
        sort!(source_events)
        #source_events = source_events[1000:min(3 * TARGET_TRAIN_LENGTH, length(source_events))]
        convert(Matrix, conditioning_events)
        conditioning_events = conditioning_events[:, 1]
	conditioning_events = conditioning_events + 1e-4 .* (rand(size(conditioning_events)[1]) .- 0.5)
        sort!(conditioning_events)
        #conditioning_events = conditioning_events[1000:min(3 * TARGET_TRAIN_LENGTH, length(conditioning_events))]

        parameters = CoTETE.CoTETEParameters(
            l_x = l_x,
            l_y = l_y,
            l_z = l_z,
            auto_find_start_and_num_events = false,
            start_event = START_OFFSET,
            num_target_events = TARGET_TRAIN_LENGTH,
            num_samples_ratio = NUM_SAMPLES_RATIO,
            k_global = K,
            num_surrogates = NUM_SURROGATES,
	    surrogate_num_samples_ratio = SURROGATE_UPSAMPLE_RATIO,
            k_perm = K_PERM,
        )

        TE, p, surrogates = CoTETE.estimate_TE_and_p_value_from_event_times(
            parameters,
            target_events,
            source_events,
            conditioning_events = [conditioning_events],
            return_surrogate_TE_values = true,
        )

        println(p)
        println(TE)
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
end
