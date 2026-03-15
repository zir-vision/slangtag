import "../filter.slang";

StructuredBuffer<uint32_t> sorted_points;
RWStructuredBuffer<uint32_t> extents_out;
RWStructuredBuffer<uint32_t> extent_count_out;
RWStructuredBuffer<uint32_t> filtered_extents_out;
RWStructuredBuffer<uint32_t> selected_extent_count_out;
RWStructuredBuffer<uint32_t> selected_point_count_out;
RWStructuredBuffer<uint32_t> point_extent_indices_out;
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
    build_blob_pair_extents(
        dispatch_thread_id.x,
        sorted_points,
        extents_out,
        extent_count_out,
        filtered_extents_out,
        selected_extent_count_out,
        selected_point_count_out,
        point_extent_indices_out,
        pc.valid_points,
        pc.tag_width,
        pc.reversed_border,
        pc.normal_border,
        pc.min_cluster_pixels,
        pc.max_cluster_pixels);
}
