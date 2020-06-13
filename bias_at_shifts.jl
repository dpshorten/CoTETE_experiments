using CSV: read
using HDF5: h5open, g_create
using Distances: Cityblock, Chebyshev, Euclidean

include("../CoTETE.jl/CoTETE.jl")

D_Y = 1
D_X = 1
D_C = 1
K = 10

REPEATS = 200
START_OFFSET = 10000
TARGET_TRAIN_LENGTH = Int(5e4)
NUM_SAMPLES_RATIO = 1
SURROGATE_UPSAMPLE_RATIO = 1.0
K_PERM = 10

GENERATION_LENGTH = START_OFFSET + 5 * TARGET_TRAIN_LENGTH
MOTHER_NOISE_STD = 5e-2
DAUGHTER_NOISE_STD = 5e-2
MOTHER_T = 1.0
DAUGHTER_GAP_1 = 0.25
DAUGHTER_GAP_2 = 0.5

BIG_SHIFT_MULTIPLIER = 300
BIG_SHIFT_BASE = 1


h5open(string("run_outputs/bias_at_shifts_test-small5.h5"), "w") do file

    shifts1 = collect(0:0.13:10)
    shifts2 = collect(-10:0.13:0)
    shifts = vcat(shifts2, shifts1)

    TE_vals = zeros(REPEATS, size(shifts, 1))
    TE_vals_surrogate = zeros(REPEATS, size(shifts, 1))
    TE_vals_shift_surrogate = zeros(REPEATS, size(shifts, 1))
    for repeat = 1:REPEATS
        println("rep ", repeat)

        mother_intervals = MOTHER_NOISE_STD .* randn(GENERATION_LENGTH) .+ MOTHER_T
        clamp!(mother_intervals, 0, 1e6)
        mother_events = cumsum(mother_intervals)
        daughter_events_1 = mother_events + DAUGHTER_NOISE_STD .* randn(GENERATION_LENGTH) .+ DAUGHTER_GAP_1
        daughter_events_2 = mother_events + DAUGHTER_NOISE_STD .* randn(GENERATION_LENGTH) .+ DAUGHTER_GAP_2
        sort!(daughter_events_1)
        sort!(daughter_events_2)
        clamp!(daughter_events_1, 0, 1e6)
        clamp!(daughter_events_2, 0, 1e6)
        conditioning_events = mother_events
        source_events = daughter_events_1
        target_events = daughter_events_2

        Threads.@threads for i = 1:size(shifts, 1)
        #for i = 1:size(shifts, 1)
            shifted_source_events = source_events .+ shifts[i]
            clamp!(shifted_source_events, 0, 1e6)
            TE = CoTETE.do_preprocessing_and_calculate_TE(
                target_events,
                shifted_source_events,
                D_X,
                D_Y,
                d_c = D_C,
                conditioning_events = conditioning_events,
                num_target_events = TARGET_TRAIN_LENGTH,
                num_samples = NUM_SAMPLES_RATIO * TARGET_TRAIN_LENGTH,
                k = K,
                start_event = START_OFFSET,
                metric = Cityblock(),
            )
            TE_vals[repeat, i] = TE

            TE_surrogate = CoTETE.do_preprocessing_and_calculate_TE(
                target_events,
                shifted_source_events,
                D_X,
                D_Y,
                d_c = D_C,
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
            TE_vals_surrogate[repeat, i] = TE_surrogate

            shifted_shifted_source_events = shifted_source_events .+ BIG_SHIFT_MULTIPLIER * (BIG_SHIFT_BASE + rand())
            TE_shift_surrogate = CoTETE.do_preprocessing_and_calculate_TE(
                target_events,
                shifted_shifted_source_events,
                D_X,
                D_Y,
                d_c = D_C,
                conditioning_events = conditioning_events,
                num_target_events = TARGET_TRAIN_LENGTH,
                num_samples = NUM_SAMPLES_RATIO * TARGET_TRAIN_LENGTH,
                k = K,
                start_event = START_OFFSET,
                metric = Cityblock(),
            )
            TE_vals_shift_surrogate[repeat, i] = TE_shift_surrogate
        end
    end
    g = g_create(file, "foo")
    g["TE"] = TE_vals
    g["TE_surrogate"] = TE_vals_surrogate
    g["TE_shift_surrogate"] = TE_vals_shift_surrogate
    g["shifts"] = shifts
end
