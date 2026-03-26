import "../filter.slang";
import "./detect-control.slang";

StructuredBuffer<uint32_t> sorted_points;
RWStructuredBuffer<uint32_t> extents_out;
RWStructuredBuffer<uint32_t> extent_count_out;
RWStructuredBuffer<uint32_t> filtered_extents_out;
RWStructuredBuffer<uint32_t> selected_extent_count_out;
RWStructuredBuffer<uint32_t> selected_point_count_out;
RWStructuredBuffer<uint32_t> selected_points_out;
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

uint32_t point_theta_from_extent_values(uint32_t min_x,
                                        uint32_t max_x,
                                        uint32_t min_y,
                                        uint32_t max_y,
                                        uint32_t x,
                                        uint32_t y)
{
    static const float kThetaScale = 8.0e6f;

    float cx = 0.5f * (float(min_x) + float(max_x)) + 0.05118f;
    float cy = 0.5f * (float(min_y) + float(max_y)) - 0.028581f;
    float theta = (atan2(float(y) - cy, float(x) - cx) + float::getPi()) * kThetaScale;
    return uint32_t(max(0.0f, theta) + 0.5f);
}

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

    BlobDiffPoint first_point = read_blob_diff_point(sorted_points, tid);
    uint32_t rep0 = first_point.rep0;
    uint32_t rep1 = first_point.rep1;
    if (tid > 0u)
    {
        BlobDiffPoint previous_point = read_blob_diff_point(sorted_points, tid - 1u);
        uint32_t previous_rep0 = previous_point.rep0;
        uint32_t previous_rep1 = previous_point.rep1;
        if (previous_rep0 == rep0 && previous_rep1 == rep1)
            return;
    }

    uint32_t min_x = first_point.x;
    uint32_t max_x = min_x;
    uint32_t min_y = first_point.y;
    uint32_t max_y = min_y;
    uint32_t count = 0u;
    int pxgx_plus_pygy_sum = 0;
    int gx_sum = 0;
    int gy_sum = 0;

    uint32_t i = tid;
    while (i < valid_points)
    {
        BlobDiffPoint point = read_blob_diff_point(sorted_points, i);
        if (point.rep0 != rep0 || point.rep1 != rep1)
            break;

        if (point.x < min_x)
            min_x = point.x;
        if (point.x > max_x)
            max_x = point.x;
        if (point.y < min_y)
            min_y = point.y;
        if (point.y > max_y)
            max_y = point.y;

        int gx;
        int gy;
        dxy_to_gradient(point.dxy, gx, gy);

        if (point.black_to_white == 0u)
        {
            gx = -gx;
            gy = -gy;
        }

        pxgx_plus_pygy_sum += int(point.x) * gx + int(point.y) * gy;
        gx_sum += gx;
        gy_sum += gy;
        count += 1u;
        i += 1u;
    }

    uint32_t extent_index;
    InterlockedAdd(extent_count_out[0], 1u, extent_index);

    BlobExtent extent;
    extent.min_x = min_x;
    extent.max_x = max_x;
    extent.min_y = min_y;
    extent.max_y = max_y;
    extent.start = tid;
    extent.count = count;
    extent.pxgx_plus_pygy_sum = pxgx_plus_pygy_sum;
    extent.gx_sum = gx_sum;
    extent.gy_sum = gy_sum;

    write_extent_word(extents_out, extent_index, kExtentMinXWord, extent.min_x);
    write_extent_word(extents_out, extent_index, kExtentMaxXWord, extent.max_x);
    write_extent_word(extents_out, extent_index, kExtentMinYWord, extent.min_y);
    write_extent_word(extents_out, extent_index, kExtentMaxYWord, extent.max_y);
    write_extent_word(extents_out, extent_index, kExtentStartingOffsetWord, extent.start);
    write_extent_word(extents_out, extent_index, kExtentCountWord, extent.count);

    // This stage now relies on per-extent initialization instead of a full
    // pre-pass clear of filtered_extents_out.
    write_extent_word(filtered_extents_out, extent_index, kExtentCountWord, 0u);

    float extent_dot = blob_extent_dot_from_moments(
        extent.pxgx_plus_pygy_sum,
        extent.gx_sum,
        extent.gy_sum,
        extent.min_x,
        extent.max_x,
        extent.min_y,
        extent.max_y);

    bool selected = select_blob_extent_from_values(
        extent.count,
        extent.min_x,
        extent.max_x,
        extent.min_y,
        extent.max_y,
        extent_dot,
        tag_width,
        reversed_border,
        normal_border,
        min_cluster_pixels,
        max_cluster_pixels);

    if (!selected)
        return;

    uint32_t selected_point_start;
    InterlockedAdd(selected_point_count_out[0], extent.count, selected_point_start);
    write_extent_word(filtered_extents_out, extent_index, kExtentMinXWord, extent.min_x);
    write_extent_word(filtered_extents_out, extent_index, kExtentMaxXWord, extent.max_x);
    write_extent_word(filtered_extents_out, extent_index, kExtentMinYWord, extent.min_y);
    write_extent_word(filtered_extents_out, extent_index, kExtentMaxYWord, extent.max_y);
    write_extent_word(filtered_extents_out, extent_index, kExtentStartingOffsetWord, selected_point_start);
    write_extent_word(filtered_extents_out, extent_index, kExtentCountWord, extent.count);
    write_extent_word(filtered_extents_out, extent_index, kExtentPxGxPyGySumWord, asuint(extent.pxgx_plus_pygy_sum));
    write_extent_word(filtered_extents_out, extent_index, kExtentGxSumWord, asuint(extent.gx_sum));
    write_extent_word(filtered_extents_out, extent_index, kExtentGySumWord, asuint(extent.gy_sum));

    for (uint32_t point_index = tid; point_index < i; ++point_index)
    {
        BlobDiffPoint source_point = read_blob_diff_point(sorted_points, point_index);
        uint32_t selected_index = selected_point_start + (point_index - tid);

        SelectedBlobPoint selected_point;
        selected_point.blob_index = extent_index;
        selected_point.theta = point_theta_from_extent_values(
            extent.min_x,
            extent.max_x,
            extent.min_y,
            extent.max_y,
            source_point.x,
            source_point.y);
        selected_point.x = source_point.x;
        selected_point.y = source_point.y;
        write_selected_blob_point(selected_points_out, selected_index, selected_point);
    }

    uint32_t selected_extent_index;
    InterlockedAdd(selected_extent_count_out[0], 1u, selected_extent_index);
}
