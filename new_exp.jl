using CSV: read
using HDF5: h5open, g_create
using Distances: Cityblock, Chebyshev, Euclidean
using Random: rand, randn
using StatsBase: sample

using CoTETE
#include("GLM_generative.jl")

l_x = 1
l_z = [1, 1, 1, 1]
l_y = 1

K = 10

START_OFFSET = 100
TARGET_TRAIN_LENGTH = Int(2e3)
#TARGET_TRAIN_LENGTH = Int(1e4)
METRIC = Cityblock()
NUM_SAMPLES_RATIO = 1.0
SURROGATE_UPSAMPLE_RATIO = 1.0
K_PERM = 10

#NUM_SURROGATES = 100
NUM_SURROGATES = 100

FOLDER = "outputs_rev_exp_corr/"

h5open("figure_8c.h5", "w") do file
    target_events = read(string(FOLDER, "x_12.dat"))
    source_events = read(string(FOLDER, "y_12.dat"))
    array_of_conditioning_events = []
    for i = 1:4
        temp = read(string(FOLDER, "z_12_n_", i,".dat"))
        push!(array_of_conditioning_events, temp)
    end

    convert(Matrix, target_events)
    target_events = target_events[:, 1]
    convert(Matrix, source_events)
    source_events = source_events[:, 1]

    new_cond = Array{Float32, 1}[]
    for events in array_of_conditioning_events
        convert(Matrix, events)
        events = events[:, 1]
        convert(Array{Float32, 1}, events)
        push!(new_cond, events)
    end

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
    )

    TE, p, surrogates = CoTETE.estimate_TE_and_p_value_from_event_times(
        parameters,
        target_events,
        source_events,
        conditioning_events = new_cond,
        return_surrogate_TE_values = true,
    )

    println(p)

    println(TE)
    sort!(surrogates)
    println(surrogates)

    # g = g_create(file, string(j, "_link_", permutation[3], permutation[1], folder))
    # g["TE"] = TE
    # g["folder"] = folder
    # surrogates = Array{Float32}(surrogates)
    # g["run"] = j
    # g["surrogates"] = surrogates
    # g["num_target_events"] = TARGET_TRAIN_LENGTH
    # g["source"] = permutation[3]
    # g["target"] = permutation[1]
end
