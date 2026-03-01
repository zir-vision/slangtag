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

void blob_diff(uint2 tid,
               Texture2D<uint8_t1> thresholded_image,
               RWStructuredBuffer<uint32_t> blobs,
               RWStructuredBuffer<uint32_t> union_markers_size,
               RWStructuredBuffer<uint32_t> result,
               uint2 dim,
               uint32_t min_blob_size)
{
    uint32_t x = tid.x;
    uint32_t y = tid.y;
    uint32_t width = dim.x;
    uint32_t height = dim.y;

    if (x >= width || y >= height || y == 0u || x == 0u || x == width - 1u || y == height - 1u)
    {
        return;
    }

    uint32_t global_input_index = x + y * width;
    uint32_t global_output_index = (x - 1u) + (y - 1u) * (width - 2u);

    uint32_t rep0 = blobs[global_input_index];
    uint32_t v0 = thresholded_image[uint2(x, y)].x;

    if (v0 == 127u || union_markers_size[rep0] < min_blob_size)
    {
        clear_blob_diff_point(result, width, height, 0u, global_output_index);
        clear_blob_diff_point(result, width, height, 1u, global_output_index);
        clear_blob_diff_point(result, width, height, 2u, global_output_index);
        clear_blob_diff_point(result, width, height, 3u, global_output_index);
        return;
    }

    uint32_t rep1_0 = blobs[global_input_index + 1u];
    uint32_t v1_0 = thresholded_image[uint2(x + 1u, y)].x;
    write_blob_diff_connection(
        result, union_markers_size, width, height, global_output_index, x, y, 0u, v0, v1_0, rep0, rep1_0, min_blob_size);

    uint32_t rep1_1 = blobs[global_input_index + width + 1u];
    uint32_t v1_1 = thresholded_image[uint2(x + 1u, y + 1u)].x;
    write_blob_diff_connection(
        result, union_markers_size, width, height, global_output_index, x, y, 1u, v0, v1_1, rep0, rep1_1, min_blob_size);

    uint32_t rep1_2 = blobs[global_input_index + width];
    uint32_t v1_2 = thresholded_image[uint2(x, y + 1u)].x;
    write_blob_diff_connection(
        result, union_markers_size, width, height, global_output_index, x, y, 2u, v0, v1_2, rep0, rep1_2, min_blob_size);

    uint32_t rep_block_left = blobs[global_input_index - 1u];
    uint32_t v1_block_left = thresholded_image[uint2(x - 1u, y)].x;
    if (v1_block_left != 127u && v1_2 != 127u && v1_2 != v1_block_left)
    {
        if (x != 1u &&
            union_markers_size[rep_block_left] >= min_blob_size &&
            union_markers_size[rep1_2] >= min_blob_size)
        {
            clear_blob_diff_point(result, width, height, 3u, global_output_index);
            return;
        }
    }

    uint32_t rep1_3 = blobs[global_input_index + width - 1u];
    uint32_t v1_3 = thresholded_image[uint2(x - 1u, y + 1u)].x;
    write_blob_diff_connection(
        result, union_markers_size, width, height, global_output_index, x, y, 3u, v0, v1_3, rep0, rep1_3, min_blob_size);
}

void blob_diff_buffer(uint2 tid,
                      StructuredBuffer<uint8_t> thresholded_image,
                      ImageSize thresholded_size,
                      RWStructuredBuffer<uint32_t> blobs,
                      RWStructuredBuffer<uint32_t> union_markers_size,
                      RWStructuredBuffer<uint32_t> result,
                      uint2 dim,
                      uint32_t min_blob_size)
{
    uint32_t x = tid.x;
    uint32_t y = tid.y;
    uint32_t width = dim.x;
    uint32_t height = dim.y;

    if (x >= width || y >= height || y == 0u || x == 0u || x == width - 1u || y == height - 1u)
    {
        return;
    }

    uint32_t global_input_index = x + y * width;
    uint32_t global_output_index = (x - 1u) + (y - 1u) * (width - 2u);

    uint32_t rep0 = blobs[global_input_index];
    uint32_t v0 = image_u8_read(thresholded_image, thresholded_size, uint2(x, y));

    if (v0 == 127u || union_markers_size[rep0] < min_blob_size)
    {
        clear_blob_diff_point(result, width, height, 0u, global_output_index);
        clear_blob_diff_point(result, width, height, 1u, global_output_index);
        clear_blob_diff_point(result, width, height, 2u, global_output_index);
        clear_blob_diff_point(result, width, height, 3u, global_output_index);
        return;
    }

    uint32_t rep1_0 = blobs[global_input_index + 1u];
    uint32_t v1_0 = image_u8_read(thresholded_image, thresholded_size, uint2(x + 1u, y));
    write_blob_diff_connection(
        result, union_markers_size, width, height, global_output_index, x, y, 0u, v0, v1_0, rep0, rep1_0, min_blob_size);

    uint32_t rep1_1 = blobs[global_input_index + width + 1u];
    uint32_t v1_1 = image_u8_read(thresholded_image, thresholded_size, uint2(x + 1u, y + 1u));
    write_blob_diff_connection(
        result, union_markers_size, width, height, global_output_index, x, y, 1u, v0, v1_1, rep0, rep1_1, min_blob_size);

    uint32_t rep1_2 = blobs[global_input_index + width];
    uint32_t v1_2 = image_u8_read(thresholded_image, thresholded_size, uint2(x, y + 1u));
    write_blob_diff_connection(
        result, union_markers_size, width, height, global_output_index, x, y, 2u, v0, v1_2, rep0, rep1_2, min_blob_size);

    uint32_t rep_block_left = blobs[global_input_index - 1u];
    uint32_t v1_block_left = image_u8_read(thresholded_image, thresholded_size, uint2(x - 1u, y));
    if (v1_block_left != 127u && v1_2 != 127u && v1_2 != v1_block_left)
    {
        if (x != 1u &&
            union_markers_size[rep_block_left] >= min_blob_size &&
            union_markers_size[rep1_2] >= min_blob_size)
        {
            clear_blob_diff_point(result, width, height, 3u, global_output_index);
            return;
        }
    }

    uint32_t rep1_3 = blobs[global_input_index + width - 1u];
    uint32_t v1_3 = image_u8_read(thresholded_image, thresholded_size, uint2(x - 1u, y + 1u));
    write_blob_diff_connection(
        result, union_markers_size, width, height, global_output_index, x, y, 3u, v0, v1_3, rep0, rep1_3, min_blob_size);
}
