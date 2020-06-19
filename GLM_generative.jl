using GLM
using Random: rand
using Distributions: Poisson
















START_AT = 500

function generate(source_events, target_events, conditioning_events, delta_t = 0.03,
                  l_x = 30, l_y = 30, l_z = 30, num_surrogates = 100)

    # source_events = source_events .- target_events[1] .+ 1.0
    # clamp!(source_events, 0, 1e6)
    # conditioning_events = conditioning_events .- target_events[1] .+ 1.0
    # clamp!(conditioning_events, 0, 1e6)
    # target_events = target_events .- target_events[1] .+ 1.0
    # clamp!(target_events, 0, 1e6)

    
    
    discretised_target_events = zeros(Int, Int(floor(target_events[end]/delta_t)) + 1)
    for i = 1:size(target_events)[1]
        discretised_target_events[Int(floor(target_events[i]/delta_t)) + 1] += 1
    end

    discretised_source_events = zeros(Int, Int(floor(source_events[end]/delta_t)) + 1)
    for i = 1:size(source_events)[1]
        discretised_source_events[Int(floor(source_events[i]/delta_t)) + 1] += 1
    end

    discretisel_zonditioning_events = zeros(Int, Int(floor(conditioning_events[end]/delta_t)) + 1)
    
    for i = 1:size(conditioning_events)[1]
        discretisel_zonditioning_events[Int(floor(conditioning_events[i]/delta_t)) + 1] += 1
    end
    
    #start_index = max(l_x, l_y, l_z) + 10
    start_index = max(l_x, l_y, l_z) + 10
    final_index = min(length(discretised_target_events), length(discretised_source_events),
                      length(discretisel_zonditioning_events))

    joint_history_representation = zeros(Float64, final_index - start_index + 1, l_x + l_y + l_z)
    for i = start_index:final_index
        joint_history_representation[i - start_index + 1, 1:l_x] =  discretised_target_events[(i - l_x):(i - 1)]
        joint_history_representation[i - start_index + 1, (l_x + 1):(l_x + l_z)] =
            discretisel_zonditioning_events[(i - l_z):(i - 1)]
        joint_history_representation[i - start_index + 1, (l_x + l_z + 1):(l_x + l_z + l_y)] =
            discretised_source_events[(i - l_y):(i - 1)]
    end
    discretised_source_events_for_fit = discretised_source_events[start_index:final_index]
    #discretised_source_events = discretised_source_events[start_index:final_index]
    #discretisel_zonditioning_events = discretisel_zonditioning_events[start_index:final_index]

    #joint_history_representation = convert(DataFrame, joint_history_representation)
    #println(DataFrames.names(joint_history_representation))

    model = fit(GeneralizedLinearModel, joint_history_representation, discretised_source_events_for_fit, Poisson())
    #println(predict(model, joint_history_representation[1:2, :]))


    new_discretised_source = discretised_source_events[:]
    for i = START_AT:final_index
        joint_history_representation = zeros(Float64, 2, l_x + l_y + l_z)
        joint_history_representation[1, 1:l_x] =  discretised_target_events[(i - l_x):(i - 1)]
        joint_history_representation[1, (l_x + 1):(l_x + l_z)] =
            discretisel_zonditioning_events[(i - l_z):(i - 1)]
        joint_history_representation[1, (l_x + l_z + 1):(l_x + l_z + l_y)] =
            new_discretised_source[(i - l_y):(i - 1)]

        num_events = predict(model, joint_history_representation)[1]
        if num_events == Inf || num_events > 5
            num_events = 5
        end
        new_discretised_source[i] = Int(round(num_events))
    end
    #println(new_discretised_source[480:520])
    #println(discretised_source_events[480:520])

    #println(new_discretised_source[1:100])
    #println(discretised_source_events[1:100])

    surrogates = []
    for k = 1:num_surrogates
        new_source = []
        for i = 1:length(new_discretised_source)
            num_events = rand(Poisson(new_discretised_source[i]))
            #num_events = Int(round(new_discretised_source[i]))
            for j = 1:num_events
                append!(new_source, (i) * delta_t + delta_t * rand())
            end
        end
        new_source = convert(Array{Float64, 1}, new_source)
        sort!(new_source)
        push!(surrogates, new_source)
    end
    return surrogates
end
