import "compute/image_buffer.slang";
static const uint32_t kBlobDiffWordsPerPoint = 6;

void write_blob_diff_point(RWStructuredBuffer<uint32_t> result,
                           uint32_t width,
                           uint32_t height,
                           uint32_t point_offset,
                           uint32_t global_output_index,
                           uint32_t rep0,
                           uint32_t rep1,
                           uint32_t x,
                           uint32_t y,
                           uint32_t dxy,
                           uint32_t black_to_white)
{
    uint32_t point_count_per_offset = (width - 2u) * (height - 2u);
    uint32_t point_index = point_count_per_offset * point_offset + global_output_index;
    uint32_t word_index = point_index * kBlobDiffWordsPerPoint;
    result[word_index + 0] = rep0;
    result[word_index + 1] = rep1;
    result[word_index + 2] = x;
    result[word_index + 3] = y;
    result[word_index + 4] = dxy;
    result[word_index + 5] = black_to_white;
}

void clear_blob_diff_point(RWStructuredBuffer<uint32_t> result,
                           uint32_t width,
                           uint32_t height,
                           uint32_t point_offset,
                           uint32_t global_output_index)
{
    write_blob_diff_point(result, width, height, point_offset, global_output_index, 0u, 0u, 0u, 0u, 0u, 0u);
}

void write_blob_diff_connection(RWStructuredBuffer<uint32_t> result,
                                RWStructuredBuffer<uint32_t> union_markers_size,
                                uint32_t width,
                                uint32_t height,
                                uint32_t global_output_index,
                                uint32_t x,
                                uint32_t y,
                                uint32_t point_offset,
                                uint32_t v0,
                                uint32_t v1,
                                uint32_t rep0,
                                uint32_t rep1,
                                uint32_t min_blob_size)
{
    if (v0 + v1 == 255u && union_markers_size[rep1] >= min_blob_size)
    {
        uint32_t lo = rep0;
        uint32_t hi = rep1;
        if (rep1 < rep0)
        {
            lo = rep1;
            hi = rep0;
        }

        write_blob_diff_point(
            result,
            width,
            height,
            point_offset,
            global_output_index,
            lo,
            hi,
            x,
            y,
            point_offset,
            v1 > v0 ? 1u : 0u);
        return;
    }

    clear_blob_diff_point(result, width, height, point_offset, global_output_index);
}

