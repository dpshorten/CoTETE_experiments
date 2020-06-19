ON_ARTEMIS = false
if ON_ARTEMIS
    import Pkg
    Pkg.instantiate()
    Pkg.add("CSV")
    Pkg.add("DelimitedFiles")
    Pkg.add("Statistics")
    Pkg.add("Random")
    Pkg.add("HDF5")
end

using CSV: read
using DelimitedFiles
using Statistics
using Random
using HDF5
using Combinatorics

function calculate_TE_discrete(
    target_events,
    source_events,
    delta_t,
    d_x,
    d_y,
    y_lag;
    c_lag = 0,
    conditioning_events = [0.0],
    d_c = 0,
)

    source_start_event = 1
    while source_events[source_start_event] < target_events[1]
        source_start_event += 1
    end
    source_end_event = source_start_event
    while source_end_event < size(source_events)[1] && source_events[source_end_event] < target_events[end]
        source_end_event += 1
    end
    source_end_event -= 1
    source_events = source_events[source_start_event:source_end_event]

    if d_c > 0
        conditioning_start_event = 1
        while conditioning_events[conditioning_start_event] < target_events[1]
            conditioning_start_event += 1
        end
        conditioning_end_event = conditioning_start_event
        while conditioning_end_event < size(conditioning_events)[1] &&
              conditioning_events[conditioning_end_event] < target_events[end]
            conditioning_end_event += 1
        end
        conditioning_end_event -= 1
        conditioning_events = conditioning_events[conditioning_start_event:conditioning_end_event]
    end

    source_events = source_events .- target_events[1] .+ 1.0
    conditioning_events = conditioning_events .- target_events[1] .+ 1.0
    target_events = target_events .- target_events[1] .+ 1.0

    discretised_target_events = zeros(Int8, Int(floor(target_events[end] / delta_t)) + 1)
    for i = 1:size(target_events)[1]
        discretised_target_events[Int(floor(target_events[i] / delta_t))+1] = 1
    end

    discretised_source_events = zeros(Int8, Int(floor(source_events[end] / delta_t)) + 1)
    for i = 1:size(source_events)[1]
        discretised_source_events[Int(floor(source_events[i] / delta_t))+1] = 1
    end

    discretised_conditioning_events = []
    if d_c > 0
        discretised_conditioning_events = zeros(Int8, Int(floor(conditioning_events[end] / delta_t)) + 1)

        for i = 1:size(conditioning_events)[1]
            discretised_conditioning_events[Int(floor(conditioning_events[i] / delta_t))+1] = 1
        end
    end

    final_index = 0
    if d_c > 0
        final_index = min(
            size(discretised_source_events)[1],
            size(discretised_target_events)[1],
            size(discretised_conditioning_events)[1],
        )
    else
        final_index = min(size(discretised_source_events)[1], size(discretised_target_events)[1])
    end
    start_index = max(d_x, d_y, d_c) + 1 + max(y_lag, c_lag)

    joint_history_representation = zeros(Int8, final_index - start_index + 1, d_x + d_y + d_c)
    target_history_representation = zeros(Int8, final_index - start_index + 1, d_x + d_c)

    for i = start_index:final_index
        joint_history_representation[i-start_index+1, 1:d_x] = discretised_target_events[(i-d_x):(i-1)]
        joint_history_representation[i-start_index+1, (d_x+1):(d_x+d_c)] =
            discretised_conditioning_events[(i-d_c-c_lag):(i-1-c_lag)]
        joint_history_representation[i-start_index+1, (d_x+d_c+1):(d_x+d_c+d_y)] =
            discretised_source_events[(i-d_y-y_lag):(i-1-y_lag)]
        target_history_representation[i-start_index+1, 1:d_x] = discretised_target_events[(i-d_x):(i-1)]
        target_history_representation[i-start_index+1, (d_x+1):(d_x+d_c)] =
            discretised_conditioning_events[(i-d_c-c_lag):(i-1-c_lag)]
    end

    discretised_target_events = discretised_target_events[start_index:final_index]
    discretised_source_events = discretised_source_events[start_index:final_index]
    if d_c > 0
        discretised_conditioning_events = discretised_conditioning_events[start_index:final_index]
    end

    histogram_target = Dict()
    if d_x > 10
        sizehint!(histogram_target, 2^(d_x + d_c - 3))
    end
    histogram_joint = Dict()
    if d_x + d_y > 10
        sizehint!(histogram_target, 2^(d_x + d_c + d_y - 3))
    end

    for i = 1:size(discretised_target_events)[1]
        ind = 1
        for j = 1:(d_x+d_c)
            ind += target_history_representation[i, j] * 2^(j - 1)
        end

        if haskey(histogram_target, ind)
            if discretised_target_events[i] == 0
                histogram_target[ind][1] += 1
            else
                histogram_target[ind][2] += 1
            end
        else
            if discretised_target_events[i] == 0
                histogram_target[ind] = [1, 0]
            else
                histogram_target[ind] = [0, 1]
            end
        end

        ind = 1
        for j = 1:(d_x+d_c+d_y)
            ind += joint_history_representation[i, j] * 2^(j - 1)
        end
        if haskey(histogram_joint, ind)
            if discretised_target_events[i] == 0
                histogram_joint[ind][1] += 1
            else
                histogram_joint[ind][2] += 1
            end
        else
            if discretised_target_events[i] == 0
                histogram_joint[ind] = [1, 0]
            else
                histogram_joint[ind] = [0, 1]
            end
        end
    end

    log_p_given_joint = 0

    for key in keys(histogram_joint)
        if histogram_joint[key][1] > 0
            log_p_given_joint +=
                histogram_joint[key][1] *
                log(histogram_joint[key][1] / (histogram_joint[key][1] + histogram_joint[key][2]))
        end
        if histogram_joint[key][2] > 0
            log_p_given_joint +=
                histogram_joint[key][2] *
                log(histogram_joint[key][2] / (histogram_joint[key][1] + histogram_joint[key][2]))
        end
    end

    log_p_given_joint = log_p_given_joint / (delta_t * size(discretised_target_events)[1])

    log_p_given_target = 0

    for key in keys(histogram_target)
        if histogram_target[key][1] > 0
            log_p_given_target +=
                histogram_target[key][1] *
                log(histogram_target[key][1] / (histogram_target[key][1] + histogram_target[key][2]))
        end
        if histogram_target[key][2] > 0
            log_p_given_target +=
                histogram_target[key][2] *
                log(histogram_target[key][2] / (histogram_target[key][1] + histogram_target[key][2]))
        end
    end

    log_p_given_target = log_p_given_target / (delta_t * size(discretised_target_events)[1])

    return log_p_given_joint - log_p_given_target

end

function bias_on_homogeneous()
    HISTORY_LENGTHS = [2, 5]
    TARGET_TRAIN_LENGTHS = [100, 1000, Int(1e4), Int(1e5)]
    REPETITIONS_PER_LENGTH = [1000, 100, 20, 20]
    DT_VALS = [1.0, 0.5, 0.2, 0.1]


    h5open(string("run_outputs/discrete_bias_at_samples", ".h5"), "w") do file
        for m = 1:length(HISTORY_LENGTHS)
            TE_vals = -100 * ones((length(DT_VALS), length(TARGET_TRAIN_LENGTHS), maximum(REPETITIONS_PER_LENGTH)))
            for l = 1:length(DT_VALS)
                println(HISTORY_LENGTHS[m], " ", DT_VALS[l])
                for i = 1:length(TARGET_TRAIN_LENGTHS)
                    Threads.@threads for j = 1:REPETITIONS_PER_LENGTH[i]
                        target_events = Float32(TARGET_TRAIN_LENGTHS[i]) * rand(TARGET_TRAIN_LENGTHS[i])
                        source_events = Float32(TARGET_TRAIN_LENGTHS[i]) * rand(TARGET_TRAIN_LENGTHS[i])
                        sort!(target_events)
                        sort!(source_events)
                        TE = calculate_TE_discrete(
                            target_events,
                            source_events,
                            DT_VALS[l],
                            HISTORY_LENGTHS[m],
                            HISTORY_LENGTHS[m],
                            0,
                        )
                        TE_vals[l, i, j] = TE
                    end
                end
            end
            g = g_create(file, string(HISTORY_LENGTHS[m]))
            g["HL"] = HISTORY_LENGTHS[m]
            g["TE"] = TE_vals
            g["num_target_events"] = TARGET_TRAIN_LENGTHS
            g["dt"] = DT_VALS
        end
    end
end

function find_lags_and_calc_TE(target_events, source_events, conditioning_events, dt, dx, dy, d_c, y_lag_max, c_lag_max)

    TE_at_c_lags = zeros(c_lag_max + 1)
    for c_lag = 0:c_lag_max
        TE_at_c_lags[c_lag+1] = calculate_TE_discrete(target_events, conditioning_events, dt, dx, d_c, c_lag, d_c = 0)
    end
    #max_TE = maximum(TE_at_c_lags)
    chosen_c_lag = findmax(TE_at_c_lags)[2] - 1
    #println("c_lag ", chosen_c_lag)

    TE_at_y_lags = zeros(c_lag_max + 1)
    for y_lag = 0:y_lag_max
        TE_at_y_lags[y_lag+1] = calculate_TE_discrete(
            target_events,
            source_events,
            dt,
            dx,
            dy,
            y_lag,
            c_lag = chosen_c_lag,
            conditioning_events = conditioning_events,
            d_c = d_c,
        )
    end
    #max_TE = maximum(TE_at_c_lags)
    return findmax(TE_at_y_lags)[1]
end

function conditional_independence_test()

    REPEATS = 10

    NUM_SURROGATES = 100
    START_OFFSET = 100
    TARGET_TRAIN_LENGTHS = [Int(5ecan4)]
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

    h5open(string("run_outputs/conditional_independence_discrete2.h5"), "w") do file
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
                        daughter_events_1 =
                            (mother_events + daughter_noise .* randn(generation_length) .+ DAUGHTER_GAP_1)
                        daughter_events_2 =
                            (mother_events + daughter_noise .* randn(generation_length) .+ DAUGHTER_GAP_2)
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
                        println("surrogate ", surrogate_vals[1], surrogate_vals[90], surrogate_vals[end])
                        println("p ", (searchsortedfirst(surrogate_vals, TE) - 1) / length(surrogate_vals))

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
end

function connectivity_test()

    REPEATS = 10

    NUM_SURROGATES = 100
    START_OFFSET = 100
    TARGET_TRAIN_LENGTH = Int(1e4)

    DT = 0.05
    MAX_LAG = 20
    d_x = 7
    d_y = 7
    d_c = 7

    h5open(string("run_outputs/connectivity_discrete_full.h5"), "w") do file
        folder = "output_stg_full17/"
        target_start_event = START_OFFSET
        target_end_event = START_OFFSET + TARGET_TRAIN_LENGTH
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

                surrogate_vals = zeros(NUM_SURROGATES)

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
                println("surrogate ", surrogate_vals[1], surrogate_vals[90], surrogate_vals[end])
                println("p ", (searchsortedfirst(surrogate_vals, TE) - 1) / length(surrogate_vals))
                println()

                g = g_create(file, string(j, "_link_", permutation[3], permutation[1], folder))
                g["TE"] = TE
                g["folder"] = folder
                surrogates = Array{Float32}(surrogate_vals)
                g["run"] = j
                g["surrogates"] = surrogates
                g["num_target_events"] = TARGET_TRAIN_LENGTH
                g["source"] = permutation[3]
                g["target"] = permutation[1]

            end
        end
    end


end

function canonical()
    TARGET_TRAIN_LENGTHS = [100, 1000, Int(1e4), Int(1e5), Int(1e6)]
    REPETITIONS_PER_LENGTH = [100, 20, 20, 20, 20]
    DT_VALS = [1.0, 0.5, 0.2, 0.1]
    OFFSET = 200

    target_events = read(string("canonical_example_dats/train_x_1.dat"))
    source_events = read(string("canonical_example_dats/train_y_1.dat"))

    convert(Matrix, target_events)
    target_events = target_events[:, 1]
    convert(Matrix, source_events)
    source_events = source_events[:, 1]

    results_TE = []

    h5open("run_outputs/connectivity_discrete.h5", "w") do file
        TE_vals = -100 * ones((length(DT_VALS), length(TARGET_TRAIN_LENGTHS), maximum(REPETITIONS_PER_LENGTH)))
        for l = 1:length(DT_VALS)
            println("dt = ", DT_VALS[l])
            for i = 1:length(TARGET_TRAIN_LENGTHS)
                println("length = ", TARGET_TRAIN_LENGTHS[i])
                for j = 1:REPETITIONS_PER_LENGTH[i]
                    target_start_event = Int(round(OFFSET + ((j - 1) * (0.5 * TARGET_TRAIN_LENGTHS[i] + 20))))
                    target_end_event = target_start_event + TARGET_TRAIN_LENGTHS[i]

                    source_start_event = 1
                    while source_events[source_start_event] < target_events[target_start_event]
                        source_start_event += 1
                    end
                    source_end_event = source_start_event
                    while source_events[source_end_event] < target_events[target_end_event]
                        source_end_event += 1
                    end
                    source_end_event -= 1

                    TE = calculate_TE_discrete(
                        target_events[target_start_event:target_end_event],
                        source_events[source_start_event:source_end_event],
                        DT_VALS[l],
                        Int(round(1 / DT_VALS[l])),
                        Int(round(1 / DT_VALS[l])),
                        0,
                    )

                    TE_vals[l, i, j] = TE
                end
            end
        end
        g = g_create(file, "foo")
        g["TE"] = TE_vals
        g["num_target_events"] = TARGET_TRAIN_LENGTHS
        g["dt"] = DT_VALS
    end

end



# function simple_test()
#     target_events = read(string("outputs_flat/easy_osc_poiss_x_", 1, ".dat"))
#     source_events = read(string("outputs_flat/easy_osc_poiss_y_", 1, "_n_", 1, ".dat"))

#     println(size(target_events))
#     println(size(source_events))

#     convert(Matrix, target_events)
#     target_events = target_events[:, 1]
#     convert(Matrix, source_events)
#     source_events = source_events[:, 1]

#     TE = 0
#     for j = 1:6
#         TE = calculate_TE_discrete(target_events[100:10100],
#                                    source_events,
#                                    1e-3, 25, 5, 0, false)
#         println(TE)
#     end


# end

#simple_test()
#connectivity_test()
#canonical()
conditional_independence_test()
#bias_on_homogeneous()
