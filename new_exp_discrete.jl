include("discretisation_testing.jl")

d_x = [2, 1, 1]
d_y = [2, 1, 1]

SIM_DT = 1e-4
DT = [8e-3, 1.6e-2, 1.6e-2]

START_OFFSET = 100
TARGET_TRAIN_LENGTHS = [Int(1e2), Int(5e2), Int(1e3)]
#TARGET_TRAIN_LENGTH = Int(1e4)

NET_SIZES = [0, 1, 2]
CONDITIONING_SIZE = [8, 16, 24]
EXTRA_TYPES = ["exc", "inh", "fake"]

NUM_SURROGATES = 100

INPUT_FOLDER = "outputs_rev_exp_corr/"

h5open(string("correlated_pop_discrete/run_", ARGS[1], ".h5"), "w") do file
    for net_size in NET_SIZES
        for extra_type in EXTRA_TYPES
            for target_length in TARGET_TRAIN_LENGTHS

                d_c = d_x[net_size + 1] .* ones(Int8, CONDITIONING_SIZE[net_size+1])

                prefix =
                    string(INPUT_FOLDER, "type_", extra_type, "_size_", net_size, "_net_", ARGS[1])

                target_events = read(string(prefix, "_x_", ".dat"))
                source_events = read(string(prefix, "_y_", ".dat"))
                array_of_conditioning_events = []
                for i = 1:CONDITIONING_SIZE[net_size+1]
                    temp = read(string(prefix, "_z__n_", i, ".dat"))
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
                    conditioning_events = array_of_conditioning_events,
                    d_c = d_c,
                )

                surrogate_vals = zeros(NUM_SURROGATES)

                Threads.@threads for j = 1:NUM_SURROGATES
                    surrogate_source_events = source_events .- 20 * (1 + rand())
                    clamp!(surrogate_source_events, 0, 1e6)

                    TE_surrogate = estimate_TE_discrete(
                        target_events[2:(target_length+2)],
                        surrogate_source_events,
                        DT[net_size + 1],
                        d_x[net_size + 1],
                        d_y[net_size + 1],
                        0;
                        c_lag = 0,
                        conditioning_events = array_of_conditioning_events,
                        d_c = d_c,
                    )


                    surrogate_vals[j] = TE_surrogate
                end

                sort!(surrogate_vals)
                println(extra_type, " ", net_size, " ", target_length)
                println("TE ", TE)
                #println("surrogate ", surrogate_vals[1], surrogate_vals[90], surrogate_vals[end])
                println(
                    "p ",
                    1 - (searchsortedfirst(surrogate_vals, TE) - 1) / length(surrogate_vals),
                )
                #println(surrogate_vals)
                println()

                # g = g_create(file, string(j, "_link_", permutation[3], permutation[1], folder))
                # g["TE"] = TE
                # g["folder"] = folder
                # surrogates = Array{Float32}(surrogate_vals)
                # g["run"] = j
                # g["surrogates"] = surrogates
                # g["num_target_events"] = TARGET_TRAIN_LENGTH
                # g["source"] = permutation[3]
                # g["target"] = permutation[1]
            end
        end
    end
end
