include("discretisation_testing.jl")



#NUM_SURROGATES = 100
NUM_SURROGATES = 100
#START_OFFSET = 100
#TARGET_TRAIN_LENGTH = Int(5e4)
TARGET_TRAIN_LENGTH = Int(1e3)

DT = 0.004
#MAX_LAG = 20
#MAX_LAG = 5
d_x = 4
d_y = 4
d_c = [4, 4, 4, 4]

FOLDER = "outputs_rev_exp_corr/"

h5open("figure_8c.h5", "w") do file
    target_events = read(string(FOLDER, "x_11.dat"))
    source_events = read(string(FOLDER, "y_11.dat"))
    array_of_conditioning_events = []
    for i = 1:4
        temp = read(string(FOLDER, "z_1_n_", i, ".dat"))
        push!(array_of_conditioning_events, temp)
    end

    convert(Matrix, target_events)
    target_events = target_events[:, 1]
    convert(Matrix, source_events)
    source_events = source_events[:, 1]
    #source_events = target_events[end] * rand(2 * length(target_events))
    #sort!(source_events)

    new_cond = Array{Float32,1}[]
    for events in array_of_conditioning_events
        convert(Matrix, events)
        events = events[:, 1]
        convert(Array{Float32,1}, events)
        push!(new_cond, events)
    end
    array_of_conditioning_events = new_cond

    TE = estimate_TE_discrete(
        target_events[2:(TARGET_TRAIN_LENGTH+2)],
        source_events,
        DT,
        d_x,
        d_y,
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
            target_events[2:(TARGET_TRAIN_LENGTH+2)],
            surrogate_source_events,
            DT,
            d_x,
            d_y,
            0;
            c_lag = 0,
            conditioning_events = array_of_conditioning_events,
            d_c = d_c,
        )


        surrogate_vals[j] = TE_surrogate
    end

    sort!(surrogate_vals)
    println("TE ", TE)
    #println("surrogate ", surrogate_vals[1], surrogate_vals[90], surrogate_vals[end])
    println("p ", 1 - (searchsortedfirst(surrogate_vals, TE) - 1) / length(surrogate_vals))
    println(surrogate_vals)
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
