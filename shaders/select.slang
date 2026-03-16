static const uint32_t kBlobDiffWordsPerPoint = 6u;
static const uint32_t kPeakWordsPerPeak = 3u;

bool blob_diff_point_is_nonzero(StructuredBuffer<uint32_t> input, uint32_t point_index)
{
    return input[point_index * kBlobDiffWordsPerPoint + 1u] != 0u;
}

bool peak_is_valid(StructuredBuffer<uint32_t> input, uint32_t peak_index)
{
    return input[peak_index * kPeakWordsPerPeak + 0u] != 0xFFFFu;
}
