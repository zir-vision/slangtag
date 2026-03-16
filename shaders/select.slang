static const uint32_t kBlobDiffWordsPerPoint = 6u;
static const uint32_t kPeakWordsPerPeak = 3u;

bool blob_diff_point_is_nonzero(StructuredBuffer<uint32_t> input, uint32_t point_index)
{
    return input[point_index * kBlobDiffWordsPerPoint + 1u] != 0u;
}

void count_nonzero_blob_diff_points(uint tid,
                                    StructuredBuffer<uint32_t> input,
                                    RWStructuredBuffer<uint32_t> count_out,
                                    uint32_t total_points)
{
    if (tid >= total_points)
        return;

    if (!blob_diff_point_is_nonzero(input, tid))
        return;

    uint32_t old_count;
    InterlockedAdd(count_out[0], 1u, old_count);
}

void filter_nonzero_blob_diff_points(uint tid,
                                     StructuredBuffer<uint32_t> input,
                                     RWStructuredBuffer<uint32_t> output,
                                     RWStructuredBuffer<uint32_t> output_count,
                                     uint32_t total_points)
{
    if (tid >= total_points)
        return;

    if (!blob_diff_point_is_nonzero(input, tid))
        return;

    uint32_t output_index;
    InterlockedAdd(output_count[0], 1u, output_index);

    uint32_t input_word_index = tid * kBlobDiffWordsPerPoint;
    uint32_t output_word_index = output_index * kBlobDiffWordsPerPoint;

    [ForceUnroll]
    for (uint32_t i = 0u; i < kBlobDiffWordsPerPoint; ++i)
    {
        output[output_word_index + i] = input[input_word_index + i];
    }
}

bool peak_is_valid(StructuredBuffer<uint32_t> input, uint32_t peak_index)
{
    return input[peak_index * kPeakWordsPerPeak + 0u] != 0xFFFFu;
}

void count_valid_peaks(uint tid,
                       StructuredBuffer<uint32_t> input,
                       RWStructuredBuffer<uint32_t> count_out,
                       uint32_t total_peaks)
{
    if (tid >= total_peaks)
        return;

    if (!peak_is_valid(input, tid))
        return;

    uint32_t old_count;
    InterlockedAdd(count_out[0], 1u, old_count);
}

void filter_valid_peaks(uint tid,
                        StructuredBuffer<uint32_t> input,
                        RWStructuredBuffer<uint32_t> output,
                        RWStructuredBuffer<uint32_t> output_count,
                        uint32_t total_peaks)
{
    if (tid >= total_peaks)
        return;

    if (!peak_is_valid(input, tid))
        return;

    uint32_t output_index;
    InterlockedAdd(output_count[0], 1u, output_index);

    uint32_t input_word_index = tid * kPeakWordsPerPeak;
    uint32_t output_word_index = output_index * kPeakWordsPerPeak;

    [ForceUnroll]
    for (uint32_t i = 0u; i < kPeakWordsPerPeak; ++i)
    {
        output[output_word_index + i] = input[input_word_index + i];
    }
}
