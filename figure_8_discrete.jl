include("discretisation_testing.jl")
using CSV: File, read
using DataFrames
using HDF5: h5open, g_create

#d_x = [3, 2, 1]
#d_y = [3, 2, 1]

#d_x = [3, 2, 1]
#d_y = [3, 2, 1]

d_x = [12, 12, 12]
d_y = [12, 12, 12]

SIM_DT = 1e-4
#DT = [7.4e-3, 1.1e-2, 2.2e-2]
#DT = [8e-3, 1.1e-2, 2.2e-2]
DT = [2e-3, 2e-3, 2e-3]

START_OFFSET = 100
TARGET_TRAIN_LENGTHS = [Int(1e2), Int(5e2), Int(1e3), Int(2e3), Int(5e3), Int(1e4)]
#TARGET_TRAIN_LENGTH = Int(1e4)

NET_SIZES = [0, 1, 2]
CONDITIONING_SIZE = [6, 12, 18]
EXTRA_TYPES = ["exc", "inh", "fake"]
#EXTRA_TYPES = ["fake_corr"]
#EXTRA_TYPES = ["fake"]

NUM_SURROGATES = 100

INPUT_FOLDER = "outputs_rev_exp_unison/"

h5open(string("uncorrelated_pop_discrete_unison_pairwise/run_", ARGS[1], ".h5"), "w") do file
    for net_size in NET_SIZES
        for extra_type in EXTRA_TYPES
            for target_length in TARGET_TRAIN_LENGTHS

                d_c = d_x[net_size + 1] .* ones(Int8, CONDITIONING_SIZE[net_size+1])

                prefix =
                    string(INPUT_FOLDER, "type_", extra_type, "_size_", net_size, "_net_", ARGS[1])

                target_events = DataFrame!(File(string(prefix, "_x_", ".dat")))
                source_events = DataFrame!(File(string(prefix, "_y_", ".dat")))
                array_of_conditioning_events = []
                for i = 1:CONDITIONING_SIZE[net_size+1]
                    temp = DataFrame!(File(string(prefix, "_z__n_", i, ".dat")))
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
                array_of_conditioning_events = new_cond

                TE = estimate_TE_discrete(
                    target_events[2:(target_length+2)],
                    source_events,
                    DT[net_size + 1],
                    d_x[net_size + 1],
                    d_y[net_size + 1],
                    0;
                    c_lag = 0,
                    #conditioning_events = array_of_conditioning_events,
                    #d_c = d_c,
                )[1]

                surrogate_vals = zeros(NUM_SURROGATES)

                Threads.@threads for j = 1:NUM_SURROGATES
                    #surrogate_source_events = source_events .- 20 * (1 + rand())
                    #clamp!(surrogate_source_events, 0, 1e6)

                    TE_surrogate = estimate_TE_discrete(
                        target_events[2:(target_length+2)],
                        source_events,
                        DT[net_size + 1],
                        d_x[net_size + 1],
                        d_y[net_size + 1],
                        0;
                        c_lag = 0,
                        #conditioning_events = array_of_conditioning_events,
                        #d_c = d_c,
                        permutation_surrogate = true,
                    )[1]


                    surrogate_vals[j] = TE_surrogate
                end

                sort!(surrogate_vals)
                println(extra_type, " ", net_size, " ", target_length)
                println("TE ", TE)
                #println("surrogate ", surrogate_vals[1], surrogate_vals[90], surrogate_vals[end])
                p = 1 - (searchsortedfirst(surrogate_vals, TE) - 1) / length(surrogate_vals)
                println(
                    "p ",
                    p
                )
                #println(surrogate_vals)
                println()

                g = g_create(file, string(net_size, extra_type, target_length))
                g["TE"] = TE
                surrogates = Array{Float32}(surrogate_vals)
                g["p"] = p
                g["net_size"] = net_size
                g["extra_type"] = extra_type
                g["target_length"] = target_length
            end
        end
    end
end
