using Distributed
@everywhere using CSV
@everywhere using HDF5: h5open, g_create
@everywhere using Distances: Cityblock, Chebyshev, Euclidean
@everywhere using Base.Filesystem: walkdir
@everywhere using CoTETE

@everywhere L_Y = 2
@everywhere L_X = 3
@everywhere K = 10

@everywhere K_PERM = 10
@everywhere NUM_SURROGATES = 100

@everywhere UPPER_INDEX = 60

@everywhere DATA_FILE_NAME = "extracted_data_wagenaar/2-2/2-2-33.2.spk"
@everywhere RESULT_FILE_NAME = "results_wagenaar/2-2/2-2-33.2-ly2lx3u.h5"

@everywhere NUM_TARGET_EVENTS_CAP = Int(2e3)

@everywhere struct EdgeRunSpec
    target_index::Int
    source_index::Int
    target_spikes::Array{AbstractFloat,1}
    source_spikes::Array{AbstractFloat,1}
end

@everywhere struct EdgeRunResult
    target_index::Int
    source_index::Int
    TE::AbstractFloat
    p::AbstractFloat
    surrogate_TE::Array{AbstractFloat}
end

@everywhere function TE_and_surrogate(jobs, results)

    while true
        run_spec = take!(jobs)
        p = 0
        TE = 0
        surrogates = zeros(NUM_SURROGATES)
        if (
            run_spec.target_index != run_spec.source_index &&
            length(run_spec.target_spikes) > 100 &&
            length(run_spec.source_spikes) > 100
        )

            parameters = CoTETE.CoTETEParameters(
                l_x = L_X,
                l_y = L_Y,
                k_global = K,
                num_target_events_cap = NUM_TARGET_EVENTS_CAP,
                num_surrogates = NUM_SURROGATES,
                transform_to_uniform = true,
                num_samples_ratio = 5.0,
                surrogate_num_samples_ratio = 5.0,
                k_perm = K_PERM,
            )

            TE, p, surrogates = CoTETE.estimate_TE_and_p_value_from_event_times(
                parameters,
                run_spec.target_spikes,
                run_spec.source_spikes,
                return_surrogate_TE_values = true,
            )
        end
        sort!(surrogates)

        temp_result = EdgeRunResult(run_spec.target_index, run_spec.source_index, TE, p, surrogates)
        # put! will only work with a tuple. Not elegant, but it works
        put!(results, (temp_result, 1))

    end
end



@everywhere function edge_producer(jobs, spikes)
    for target_index = 1:UPPER_INDEX
        for source_index = 1:UPPER_INDEX
            temp_run_spec = EdgeRunSpec(
                target_index,
                source_index,
                deepcopy(spikes[target_index]),
                deepcopy(spikes[source_index]),
            )
            put!(jobs, temp_run_spec)
        end
    end
end

f = open(DATA_FILE_NAME)
lines = readlines(f)
spikes = []
for line in lines
    temp = Float64[]
    times = split(line, ",")
    for time in times
        if length(time) > 0
            push!(temp, parse(Float64, time))
        end
    end
    if (length(temp) > 0)
        temp = temp .- temp[1] .+ 1
    end
    temp = temp + (rand(length(temp)) .- 0.5)
    sort!(temp)
    push!(spikes, temp[10:(end-10)])
end


#println(root, " ", dirs, " ", files)
taskref = Ref{Task}()
jobs = RemoteChannel((taskref = taskref) -> Channel{EdgeRunSpec}(100))
results = RemoteChannel(() -> Channel{Any}(100))

# Create the producer
remote_do(edge_producer, 2, jobs, spikes)
# Create the consumers
for i = 3:nworkers()
    remote_do(TE_and_surrogate, i, jobs, results)
end


for i = 1:(UPPER_INDEX^2)
    run_result, where = take!(results)
    println(run_result.target_index, " ", run_result.source_index)
    h5open(RESULT_FILE_NAME, isfile(RESULT_FILE_NAME) ? "r+" : "w") do file
        g = g_create(file, string(run_result.target_index, " ", run_result.source_index))
        g["TE"] = run_result.TE
        surrogates = Array{Float32}(run_result.surrogate_TE)
        g["p"] = run_result.p
        g["surrogates"] = surrogates
        g["target_index"] = run_result.target_index
        g["source_index"] = run_result.source_index
    end
end

rmprocs()
