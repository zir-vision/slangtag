import "../filter.slang";
import "./detect-control.slang";

StructuredBuffer<uint32_t> sorted_points;
RWStructuredBuffer<uint32_t> extents_out;
RWStructuredBuffer<uint32_t> extent_count_out;
RWStructuredBuffer<uint32_t> filtered_extents_out;
RWStructuredBuffer<uint32_t> selected_extent_count_out;
RWStructuredBuffer<uint32_t> selected_point_count_out;
RWStructuredBuffer<uint32_t> point_extent_indices_out;
StructuredBuffer<uint32_t> control;
struct PushConstants
{
    uint32_t valid_points;
    uint32_t tag_width;
    uint32_t reversed_border;
    uint32_t normal_border;
    uint32_t min_cluster_pixels;
    uint32_t max_cluster_pixels;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint32_t tid = dispatch_thread_id.x;
    uint32_t valid_points = control[kControlBlobDiffFilteredCountWord];
    uint32_t tag_width = pc.tag_width;
    uint32_t reversed_border = pc.reversed_border;
    uint32_t normal_border = pc.normal_border;
    uint32_t min_cluster_pixels = pc.min_cluster_pixels;
    uint32_t max_cluster_pixels = pc.max_cluster_pixels;
    if (tid >= valid_points)
        return;

    uint32_t rep0 = read_blob_diff_point_word(sorted_points, tid, kPointRep0Word);
    uint32_t rep1 = read_blob_diff_point_word(sorted_points, tid, kPointRep1Word);
    if (tid > 0u)
    {
        uint32_t previous_rep0 = read_blob_diff_point_word(sorted_points, tid - 1u, kPointRep0Word);
        uint32_t previous_rep1 = read_blob_diff_point_word(sorted_points, tid - 1u, kPointRep1Word);
        if (previous_rep0 == rep0 && previous_rep1 == rep1)
            return;
    }

    uint32_t min_x = read_blob_diff_point_word(sorted_points, tid, kPointXWord);
    uint32_t max_x = min_x;
    uint32_t min_y = read_blob_diff_point_word(sorted_points, tid, kPointYWord);
    uint32_t max_y = min_y;
    uint32_t count = 0u;
    int pxgx_plus_pygy_sum = 0;
    int gx_sum = 0;
    int gy_sum = 0;

    uint32_t i = tid;
    while (i < valid_points)
    {
        uint32_t point_rep0 = read_blob_diff_point_word(sorted_points, i, kPointRep0Word);
        uint32_t point_rep1 = read_blob_diff_point_word(sorted_points, i, kPointRep1Word);
        if (point_rep0 != rep0 || point_rep1 != rep1)
            break;

        uint32_t x = read_blob_diff_point_word(sorted_points, i, kPointXWord);
        uint32_t y = read_blob_diff_point_word(sorted_points, i, kPointYWord);
        if (x < min_x)
            min_x = x;
        if (x > max_x)
            max_x = x;
        if (y < min_y)
            min_y = y;
        if (y > max_y)
            max_y = y;

        int gx;
        int gy;
        dxy_to_gradient(read_blob_diff_point_word(sorted_points, i, kPointDxyWord), gx, gy);

        if (read_blob_diff_point_word(sorted_points, i, kPointBlackToWhiteWord) == 0u)
        {
            gx = -gx;
            gy = -gy;
        }

        pxgx_plus_pygy_sum += int(x) * gx + int(y) * gy;
        gx_sum += gx;
        gy_sum += gy;
        count += 1u;
        i += 1u;
    }

    uint32_t extent_index;
    InterlockedAdd(extent_count_out[0], 1u, extent_index);

    for (uint32_t point_index = tid; point_index < i; ++point_index)
    {
        point_extent_indices_out[point_index] = extent_index;
    }

    write_extent_word(extents_out, extent_index, kExtentMinXWord, min_x);
    write_extent_word(extents_out, extent_index, kExtentMaxXWord, max_x);
    write_extent_word(extents_out, extent_index, kExtentMinYWord, min_y);
    write_extent_word(extents_out, extent_index, kExtentMaxYWord, max_y);
    write_extent_word(extents_out, extent_index, kExtentStartingOffsetWord, tid);
    write_extent_word(extents_out, extent_index, kExtentCountWord, count);

    float extent_dot = blob_extent_dot_from_moments(
        pxgx_plus_pygy_sum,
        gx_sum,
        gy_sum,
        min_x,
        max_x,
        min_y,
        max_y);

    bool selected = select_blob_extent_from_values(
        count,
        min_x,
        max_x,
        min_y,
        max_y,
        extent_dot,
        tag_width,
        reversed_border,
        normal_border,
        min_cluster_pixels,
        max_cluster_pixels);

    if (!selected)
        return;

    uint32_t selected_point_start;
    InterlockedAdd(selected_point_count_out[0], count, selected_point_start);
    write_extent_word(filtered_extents_out, extent_index, kExtentMinXWord, min_x);
    write_extent_word(filtered_extents_out, extent_index, kExtentMaxXWord, max_x);
    write_extent_word(filtered_extents_out, extent_index, kExtentMinYWord, min_y);
    write_extent_word(filtered_extents_out, extent_index, kExtentMaxYWord, max_y);
    write_extent_word(filtered_extents_out, extent_index, kExtentStartingOffsetWord, selected_point_start);
    write_extent_word(filtered_extents_out, extent_index, kExtentCountWord, count);
    write_extent_word(filtered_extents_out, extent_index, kExtentPxGxPyGySumWord, asuint(pxgx_plus_pygy_sum));
    write_extent_word(filtered_extents_out, extent_index, kExtentGxSumWord, asuint(gx_sum));
    write_extent_word(filtered_extents_out, extent_index, kExtentGySumWord, asuint(gy_sum));

    uint32_t selected_extent_index;
    InterlockedAdd(selected_extent_count_out[0], 1u, selected_extent_index);
}
