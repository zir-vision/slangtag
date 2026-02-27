static const uint32_t kBlobDiffWordsPerPoint = 6u;
static const uint32_t kSelectedBlobPointWordsPerPoint = 4u;
static const uint32_t kPeakWordsPerPeak = 3u;

bool blob_diff_key_less(uint32_t rep0_a,
                        uint32_t rep1_a,
                        uint32_t rep0_b,
                        uint32_t rep1_b)
{
    // Match QuadBoundaryPoint::rep01 ordering: high bits are rep1, low bits are rep0.
    if (rep1_a != rep1_b)
    {
        return rep1_a < rep1_b;
    }

    return rep0_a < rep0_b;
}

void prepare_blob_diff_points(uint tid,
                              StructuredBuffer<uint32_t> input,
                              RWStructuredBuffer<uint32_t> output,
                              uint32_t valid_points,
                              uint32_t total_points)
{
    if (tid >= total_points)
        return;

    uint32_t output_word_index = tid * kBlobDiffWordsPerPoint;

    if (tid < valid_points)
    {
        uint32_t input_word_index = output_word_index;

        [ForceUnroll]
        for (uint32_t i = 0u; i < kBlobDiffWordsPerPoint; ++i)
        {
            output[output_word_index + i] = input[input_word_index + i];
        }
        return;
    }

    // Sentinel points sort to the end of ascending order.
    output[output_word_index + 0u] = 0xFFFFFFFFu;
    output[output_word_index + 1u] = 0xFFFFFFFFu;
    output[output_word_index + 2u] = 0u;
    output[output_word_index + 3u] = 0u;
    output[output_word_index + 4u] = 0u;
    output[output_word_index + 5u] = 0u;
}

void bitonic_sort_blob_diff_points(uint tid,
                                   RWStructuredBuffer<uint32_t> points,
                                   uint32_t total_points,
                                   uint32_t j,
                                   uint32_t k)
{
    if (tid >= total_points)
        return;

    uint32_t ixj = tid ^ j;
    if (ixj <= tid || ixj >= total_points)
        return;

    uint32_t word_index_a = tid * kBlobDiffWordsPerPoint;
    uint32_t word_index_b = ixj * kBlobDiffWordsPerPoint;

    uint32_t rep0_a = points[word_index_a + 0u];
    uint32_t rep1_a = points[word_index_a + 1u];
    uint32_t rep0_b = points[word_index_b + 0u];
    uint32_t rep1_b = points[word_index_b + 1u];

    bool a_less_than_b = blob_diff_key_less(rep0_a, rep1_a, rep0_b, rep1_b);
    bool b_less_than_a = blob_diff_key_less(rep0_b, rep1_b, rep0_a, rep1_a);

    bool ascending = (tid & k) == 0u;
    bool should_swap = ascending ? b_less_than_a : a_less_than_b;

    if (!should_swap)
        return;

    [ForceUnroll]
    for (uint32_t i = 0u; i < kBlobDiffWordsPerPoint; ++i)
    {
        uint32_t tmp = points[word_index_a + i];
        points[word_index_a + i] = points[word_index_b + i];
        points[word_index_b + i] = tmp;
    }
}

bool selected_blob_point_key_less(uint32_t blob_index_a,
                                  uint32_t theta_a,
                                  uint32_t blob_index_b,
                                  uint32_t theta_b)
{
    if (blob_index_a != blob_index_b)
        return blob_index_a < blob_index_b;

    return theta_a < theta_b;
}

void prepare_selected_blob_points(uint tid,
                                  StructuredBuffer<uint32_t> input,
                                  RWStructuredBuffer<uint32_t> output,
                                  uint32_t valid_points,
                                  uint32_t total_points)
{
    if (tid >= total_points)
        return;

    uint32_t output_word_index = tid * kSelectedBlobPointWordsPerPoint;

    if (tid < valid_points)
    {
        uint32_t input_word_index = output_word_index;

        [ForceUnroll]
        for (uint32_t i = 0u; i < kSelectedBlobPointWordsPerPoint; ++i)
        {
            output[output_word_index + i] = input[input_word_index + i];
        }
        return;
    }

    output[output_word_index + 0u] = 0xFFFFFFFFu;
    output[output_word_index + 1u] = 0xFFFFFFFFu;
    output[output_word_index + 2u] = 0u;
    output[output_word_index + 3u] = 0u;
}

void bitonic_sort_selected_blob_points(uint tid,
                                       RWStructuredBuffer<uint32_t> points,
                                       uint32_t total_points,
                                       uint32_t j,
                                       uint32_t k)
{
    if (tid >= total_points)
        return;

    uint32_t ixj = tid ^ j;
    if (ixj <= tid || ixj >= total_points)
        return;

    uint32_t word_index_a = tid * kSelectedBlobPointWordsPerPoint;
    uint32_t word_index_b = ixj * kSelectedBlobPointWordsPerPoint;

    uint32_t blob_index_a = points[word_index_a + 0u];
    uint32_t theta_a = points[word_index_a + 1u];
    uint32_t blob_index_b = points[word_index_b + 0u];
    uint32_t theta_b = points[word_index_b + 1u];

    bool a_less_than_b = selected_blob_point_key_less(
        blob_index_a,
        theta_a,
        blob_index_b,
        theta_b);
    bool b_less_than_a = selected_blob_point_key_less(
        blob_index_b,
        theta_b,
        blob_index_a,
        theta_a);

    bool ascending = (tid & k) == 0u;
    bool should_swap = ascending ? b_less_than_a : a_less_than_b;

    if (!should_swap)
        return;

    [ForceUnroll]
    for (uint32_t i = 0u; i < kSelectedBlobPointWordsPerPoint; ++i)
    {
        uint32_t tmp = points[word_index_a + i];
        points[word_index_a + i] = points[word_index_b + i];
        points[word_index_b + i] = tmp;
    }
}

bool peak_key_less(uint32_t blob_index_a,
                   uint32_t error_word_a,
                   uint32_t blob_index_b,
                   uint32_t error_word_b)
{
    if (blob_index_a != blob_index_b)
        return blob_index_a < blob_index_b;

    return asfloat(error_word_a) < asfloat(error_word_b);
}

void prepare_peaks(uint tid,
                   StructuredBuffer<uint32_t> input,
                   RWStructuredBuffer<uint32_t> output,
                   uint32_t valid_peaks,
                   uint32_t total_peaks)
{
    if (tid >= total_peaks)
        return;

    uint32_t output_word_index = tid * kPeakWordsPerPeak;
    if (tid < valid_peaks)
    {
        uint32_t input_word_index = output_word_index;
        [ForceUnroll]
        for (uint32_t i = 0u; i < kPeakWordsPerPeak; ++i)
        {
            output[output_word_index + i] = input[input_word_index + i];
        }
        return;
    }

    output[output_word_index + 0u] = 0xFFFFFFFFu;
    output[output_word_index + 1u] = asuint(0.0f);
    output[output_word_index + 2u] = 0u;
}

void bitonic_sort_peaks(uint tid,
                        RWStructuredBuffer<uint32_t> peaks,
                        uint32_t total_peaks,
                        uint32_t j,
                        uint32_t k)
{
    if (tid >= total_peaks)
        return;

    uint32_t ixj = tid ^ j;
    if (ixj <= tid || ixj >= total_peaks)
        return;

    uint32_t word_index_a = tid * kPeakWordsPerPeak;
    uint32_t word_index_b = ixj * kPeakWordsPerPeak;

    uint32_t blob_index_a = peaks[word_index_a + 0u];
    uint32_t error_word_a = peaks[word_index_a + 1u];
    uint32_t blob_index_b = peaks[word_index_b + 0u];
    uint32_t error_word_b = peaks[word_index_b + 1u];

    bool a_less_than_b = peak_key_less(blob_index_a, error_word_a, blob_index_b, error_word_b);
    bool b_less_than_a = peak_key_less(blob_index_b, error_word_b, blob_index_a, error_word_a);

    bool ascending = (tid & k) == 0u;
    bool should_swap = ascending ? b_less_than_a : a_less_than_b;
    if (!should_swap)
        return;

    [ForceUnroll]
    for (uint32_t i = 0u; i < kPeakWordsPerPeak; ++i)
    {
        uint32_t tmp = peaks[word_index_a + i];
        peaks[word_index_a + i] = peaks[word_index_b + i];
        peaks[word_index_b + i] = tmp;
    }
}
