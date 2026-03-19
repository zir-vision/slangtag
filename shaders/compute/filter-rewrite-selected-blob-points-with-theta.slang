import "../filter.slang";
import "./detect-control.slang";

StructuredBuffer<uint32_t> sorted_points;
StructuredBuffer<uint32_t> extents_in;
StructuredBuffer<uint32_t> point_extent_indices;
StructuredBuffer<uint32_t> filtered_extents;
RWStructuredBuffer<uint32_t> selected_points_out;
StructuredBuffer<uint32_t> control;
struct PushConstants
{
    uint32_t extent_count;
    uint32_t valid_points;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

uint32_t point_theta_from_extent(StructuredBuffer<uint32_t> extents,
                                 uint32_t extent_index,
                                 uint32_t x,
                                 uint32_t y)
{
    static const float kThetaScale = 8.0e6f;

    float cx = blob_extent_cx(extents, extent_index);
    float cy = blob_extent_cy(extents, extent_index);

    float theta = (atan2(float(y) - cy, float(x) - cx) + float::getPi()) * kThetaScale;
    return uint32_t(max(0.0f, theta) + 0.5f);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint32_t tid = dispatch_thread_id.x;
    uint32_t extent_count = control[kControlBlobExtentCountWord];
    uint32_t valid_points = control[kControlBlobDiffFilteredCountWord];
    if (tid >= valid_points)
        return;

    uint32_t extent_index = point_extent_indices[tid];
    if (extent_index >= extent_count)
        return;

    uint32_t selected_count = read_extent_word(filtered_extents, extent_index, kExtentCountWord);
    if (selected_count == 0u)
        return;

    uint32_t original_start = read_extent_word(extents_in, extent_index, kExtentStartingOffsetWord);
    uint32_t local_offset = tid - original_start;
    if (local_offset >= selected_count)
        return;

    uint32_t selected_start = read_extent_word(filtered_extents, extent_index, kExtentStartingOffsetWord);
    uint32_t selected_index = selected_start + local_offset;

    uint32_t x = read_blob_diff_point_word(sorted_points, tid, kPointXWord);
    uint32_t y = read_blob_diff_point_word(sorted_points, tid, kPointYWord);

    write_selected_blob_point_word(
        selected_points_out,
        selected_index,
        kSelectedPointBlobIndexWord,
        extent_index);
    write_selected_blob_point_word(
        selected_points_out,
        selected_index,
        kSelectedPointThetaWord,
        point_theta_from_extent(extents_in, extent_index, x, y));
    write_selected_blob_point_word(
        selected_points_out,
        selected_index,
        kSelectedPointXWord,
        x);
    write_selected_blob_point_word(
        selected_points_out,
        selected_index,
        kSelectedPointYWord,
        y);
}
