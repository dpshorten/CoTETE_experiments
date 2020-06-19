using CSV: read
using DelimitedFiles
using Statistics
using HDF5
using Combinatorics
using Distances: Chebyshev, Euclidean, Cityblock, evaluate, colwise, Minkowski, Metric
using Random
using SpecialFunctions: digamma, gamma
using StatsFuns

include("../CoTETE.jl/NearestNeighbors.jl/src/NearestNeighbors.jl")
include("../CoTETE.jl/CoTETE.jl")

D = 2
d_x = 1
K = 2
START_AFTER_event = 700
MAX_MAX_RATE = 500
INITIAL_MAX_RATE = 100
FOLDER = "output_stg_full12/"
TARGET = "lp"
CONDITIONAL_1 = "py"
CONDITIONAL_2 = "abpd"
RUNS_PER_THREAD = 5
NUM_FILES = 5
NUM_GENERATED_eventS = 40000
#NUM_GENERATED_eventS = 1000

for file_num = 1:NUM_FILES

    target = read(string(FOLDER, TARGET, "_", file_num, ".dat"))
    conditional_1 = read(string(FOLDER, CONDITIONAL_1, "_", file_num, ".dat"))
    conditional_2 = read(string(FOLDER, CONDITIONAL_1, "_", file_num, ".dat"))


    convert(Matrix, conditional_1)
    conditional_1 = conditional_1[:, 1]

    convert(Matrix, conditional_2)
    conditional_2 = conditional_2[:, 1]

    convert(Matrix, target)
    target = target[:, 1]

    representation_joint, representation_conditionals,
    sampled_representation_joint, sampled_representation_conditionals,
    target_event_times, target_history_start_times,
    sampled_event_times, sampled_history_start_times =
        CoTETE.construct_history_embeddings(target, conditional_1,
                                          d_x, D, num_target_events = Int(2e4),
                                          num_samples = Int(2e4), start_event = 500,
                                          conditioning_events = conditional_2, d_c = D)

    combined = []
    push!(combined, representation_joint)
    push!(combined, sampled_representation_joint)
    combined = hcat(combined...)

    sorted_combined = sort!(combined, dims = 2)

    w = CoTETE.nataf_transform!(representation_joint, sampled_representation_joint)

    tree_joint = NearestNeighbors.KDTree(representation_joint, Cityblock(), reorder = false, leafsize = 10)
    tree_sampled_joint = NearestNeighbors.KDTree(sampled_representation_joint, Cityblock(), reorder = false, leafsize = 10)

    average_rate = length(target)/target[end]
    target_orig = copy(target)


    for j = 1:RUNS_PER_THREAD

        println(file_num, " ", j)
        
        max_rate = INITIAL_MAX_RATE
        target = zeros(Float64, START_AFTER_event)
        target[:] = target_orig[1:START_AFTER_event]
        current_time = target[end]
        event_time_arrays = [target, conditional_2, conditional_1]
        trackers = ones(Integer, length(event_time_arrays))
        while length(event_time_arrays[1]) < NUM_GENERATED_eventS
            
            current_time = current_time + (1/max_rate)Random.randexp()
            trackers[1] = length(event_time_arrays[1])
            for i = 2:length(trackers)
                while (event_time_arrays[i][trackers[i] + 1] <  current_time)
                    trackers[i] += 1
                end
            end
            
            embedding, start_time = CoTETE.make_one_embedding(current_time,
                                                            event_time_arrays,
                                                            trackers,
                                                            [d_x, D, D])
            orig_embedding = copy(embedding)

            for i = 1:size(embedding, 1)
                embedding[i] = searchsortedfirst(sorted_combined[i, :], embedding[i])/(size(sorted_combined, 2) + 1)
                embedding[i] = StatsFuns.norminvcdf.(embedding[i])
            end
            embedding = w * embedding
            embedding = convert(Array{Float64, 1}, embedding)
            
            
            indices_joint, radii_joint = NearestNeighbors.knn(tree_joint, embedding,
                                                              current_time + 1, start_time - 1,
                                                              target_event_times, target_history_start_times, 
                                                              K)
            p_joint = 1/(maximum(radii_joint)^(d_x + 2 * D))

            indices_sampled_joint, radii_sampled_joint = NearestNeighbors.knn(tree_sampled_joint, embedding,
                                                                              current_time + 1, start_time - 1,
                                                                              sampled_event_times, sampled_history_start_times, 
                                                                              K)
            p_sampled_joint = 1/(maximum(radii_sampled_joint)^(d_x + 2 * D))

            p = (average_rate * p_joint)/p_sampled_joint
            if isnan(p)
                p = 1
            end
            #if p > max_rate
            #   println("Max rate too low")
            #end
            if Random.rand() < p/max_rate || ((current_time - event_time_arrays[1][end]) > 2)
                push!(event_time_arrays[1], current_time)
            end
            max_rate = min(2 * p + 100, MAX_MAX_RATE)
        end

        writedlm(string("output_stg_full12_surrogates/", TARGET, "_", file_num, "_surrogate_",
                        (parse(Int, ARGS[1]) - 1) * RUNS_PER_THREAD + j, ".dat"), event_time_arrays[1], "\n")
    end
end































