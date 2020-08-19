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

function estimate_TE_discrete(
    target_events,
    source_events,
    delta_t,
    d_x,
    d_y,
    y_lag;
    c_lag = 0,
    conditioning_events = [[]],
    d_c = [0],
)

    target_events = deepcopy(target_events)
    source_events = deepcopy(source_events)
    conditioning_events = deepcopy(conditioning_events)

    source_start_event = 1
    while source_events[source_start_event] < target_events[1]
        source_start_event += 1
    end
    source_end_event = source_start_event
    while source_end_event < size(source_events)[1] &&
        source_events[source_end_event] < target_events[end]
        source_end_event += 1
    end
    source_end_event -= 1
    source_events = source_events[source_start_event:source_end_event]

    if d_c[1] != 0
        for i = 1:length(d_c)
            conditioning_start_event = 1
            #println(length(conditioning_events[i]))
            while conditioning_events[i][conditioning_start_event] < target_events[1]
                conditioning_start_event += 1
            end
            conditioning_end_event = conditioning_start_event
            while conditioning_end_event < size(conditioning_events[i], 1) &&
                conditioning_events[i][conditioning_end_event] < target_events[end]
                conditioning_end_event += 1
            end
            conditioning_end_event -= 1
            conditioning_events[i] =
                conditioning_events[i][conditioning_start_event:conditioning_end_event]
        end
    end

    source_events = source_events .- target_events[1] .+ 1.0
    for i = 1:length(d_c)
        conditioning_events[i] = conditioning_events[i] .- target_events[1] .+ 1.0
    end
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
    if d_c[1] != 0
        for i = 1:length(d_c)
            temp = zeros(Int8, Int(floor(conditioning_events[i][end] / delta_t)) + 1)

            for j = 1:size(conditioning_events)[1]
                temp[Int(floor(conditioning_events[i][j] / delta_t))+1] = 1
            end
            push!(discretised_conditioning_events, temp)
        end
    end

    final_index = 0
    if d_c[1] != 0 > 0
        conditioning_lengths = []
        for i = 1:length(d_c)
            push!(conditioning_lengths, size(discretised_conditioning_events[i], 1))
        end
        final_index = min(
            size(discretised_source_events)[1],
            size(discretised_target_events)[1],
            minimum(conditioning_lengths),
        )
    else
        final_index = min(size(discretised_source_events)[1], size(discretised_target_events)[1])
    end
    start_index = max(d_x, d_y, maximum(d_c)) + 1 + max(y_lag, maximum(c_lag))

    joint_history_representation = zeros(Int8, final_index - start_index + 1, d_x + d_y + sum(d_c))
    target_history_representation = zeros(Int8, final_index - start_index + 1, d_x + sum(d_c))

    for i = start_index:final_index
        joint_history_representation[i-start_index+1, 1:d_x] =
            discretised_target_events[(i-d_x):(i-1)]
        if d_c[1] != 0 > 0
            for j = 1:length(d_c)
                joint_history_representation[
                    i-start_index+1,
                    (d_x+sum(d_c[1:(j-1)])+1):(d_x+sum(d_c[1:j])),
                ] = discretised_conditioning_events[j][(i-d_c[j]-c_lag):(i-1-c_lag)]
            end
        end
        joint_history_representation[i-start_index+1, (d_x+sum(d_c)+1):(d_x+sum(d_c)+d_y)] =
            discretised_source_events[(i-d_y-y_lag):(i-1-y_lag)]
        target_history_representation[i-start_index+1, 1:d_x] =
            discretised_target_events[(i-d_x):(i-1)]
        if d_c[1] != 0 > 0
            for j = 1:length(d_c)
                target_history_representation[
                    i-start_index+1,
                    (d_x+sum(d_c[1:(j-1)])+1):(d_x+sum(d_c[1:j])),
                ] = discretised_conditioning_events[j][(i-d_c[j]-c_lag):(i-1-c_lag)]
            end
        end
    end

    discretised_target_events = discretised_target_events[start_index:final_index]
    discretised_source_events = discretised_source_events[start_index:final_index]
    if d_c[1] != 0 > 0
        for i = 1:length(d_c)
            discretised_conditioning_events[i] =
                discretised_conditioning_events[i][start_index:final_index]
        end
    end

    histogram_target = Dict()
    if d_x > 10
        sizehint!(histogram_target, 2^(d_x + sum(d_c) - 3))
    end
    histogram_joint = Dict()
    if d_x + d_y > 10
        sizehint!(histogram_joint, 2^(d_x + sum(d_c) + d_y - 3))
    end

    for i = 1:size(discretised_target_events)[1]
        ind = 1
        for j = 1:(d_x+sum(d_c))
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
        for j = 1:(d_x+sum(d_c)+d_y)
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
                histogram_target[key][1] * log(
                    histogram_target[key][1] /
                    (histogram_target[key][1] + histogram_target[key][2]),
                )
        end
        if histogram_target[key][2] > 0
            log_p_given_target +=
                histogram_target[key][2] * log(
                    histogram_target[key][2] /
                    (histogram_target[key][1] + histogram_target[key][2]),
                )
        end
    end

    log_p_given_target = log_p_given_target / (delta_t * size(discretised_target_events)[1])

    return log_p_given_joint - log_p_given_target

end



function find_lags_and_calc_TE(
    target_events,
    source_events,
    conditioning_events,
    dt,
    dx,
    dy,
    d_c,
    y_lag_max,
    c_lag_max,
)

    TE_at_c_lags = zeros(c_lag_max + 1)
    for c_lag = 0:c_lag_max
        TE_at_c_lags[c_lag+1] =
            estimate_TE_discrete(target_events, conditioning_events, dt, dx, d_c, c_lag, d_c = 0)
    end
    #max_TE = maximum(TE_at_c_lags)
    chosen_c_lag = findmax(TE_at_c_lags)[2] - 1
    #println("c_lag ", chosen_c_lag)

    TE_at_y_lags = zeros(c_lag_max + 1)
    for y_lag = 0:y_lag_max
        TE_at_y_lags[y_lag+1] = estimate_TE_discrete(
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
