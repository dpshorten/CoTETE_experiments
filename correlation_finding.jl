using CSV: read
using HDF5: h5open, g_create
using Distances: Cityblock, Chebyshev, Euclidean
using Combinatorics: permutations
using Random: rand, randn
using StatsBase: sample
using PyCall


include("../CoTETE.jl/CoTETE.jl")
#include("GLM_generative.jl")

d_x = 2
d_c = 2
d_y = 2

K = 10

START_OFFSET = 200
TARGET_TRAIN_LENGTH = Int(4e4)
METRIC = Cityblock()
NUM_SAMPLES_RATIO = 1
SURROGATE_UPSAMPLE_RATIO = 1.0
K_PERM = 5

NUM_SURROGATES = 100

FOLDERS = ["output_stg_full_spk/"]

h5open("run_outputs/stg_foo_bar_4_min.h5", "w") do file
    f = open("out", "w")
    for folder in FOLDERS
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
                #conditioning_events = conditioning_events[1000:min(3 * TARGET_TRAIN_LENGTH, length(conditioning_events))]
                TE = CoTETE.do_preprocessing_and_calculate_TE(
                    target_events,
                    source_events,
                    d_x,
                    d_y,
                    d_c = d_c,
                    conditioning_events = conditioning_events,
                    num_target_events = TARGET_TRAIN_LENGTH,
                    num_samples = NUM_SAMPLES_RATIO * TARGET_TRAIN_LENGTH,
                    k = K,
                    start_event = START_OFFSET,
                    metric = Cityblock(),
                )

                println("TE ", TE)
                surrogates = zeros(Float64, NUM_SURROGATES)
                #source_surrogates = generate(source_events, target_events, conditioning_events)
                Threads.@threads for i = 1:NUM_SURROGATES
                    #for i = 1:1
                    source_events_surrogate = copy(source_events)
                    TE_surrogate = CoTETE.do_preprocessing_and_calculate_TE(
                        target_events,
                        source_events,
                        d_x,
                        d_y,
                        d_c = d_c,
                        conditioning_events = conditioning_events,
                        num_target_events = TARGET_TRAIN_LENGTH,
                        num_samples = NUM_SAMPLES_RATIO * TARGET_TRAIN_LENGTH,
                        k = K,
                        start_event = START_OFFSET,
                        metric = Cityblock(),
                        is_surrogate = true,
                        surrogate_upsample_ratio = SURROGATE_UPSAMPLE_RATIO,
                        k_perm = K_PERM,
                    )
                    surrogates[i] = TE_surrogate
                end
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
end
