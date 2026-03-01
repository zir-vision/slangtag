import "../filter.slang";

StructuredBuffer<uint32_t> extents_in;
RWStructuredBuffer<uint32_t> extents_out;
RWStructuredBuffer<uint32_t> selected_extent_count_out;
RWStructuredBuffer<uint32_t> selected_point_count_out;
uint32_t extent_count;
uint32_t tag_width;
uint32_t reversed_border;
uint32_t normal_border;
uint32_t min_cluster_pixels;
uint32_t max_cluster_pixels;

[shader("compute")]
[numthreads(1, 1, 1)]
void filter_blob_pair_extents(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    filter_blob_pair_extents(
        dispatch_thread_id.x,
        extents_in,
        extents_out,
        selected_extent_count_out,
        selected_point_count_out,
        extent_count,
        tag_width,
        reversed_border,
        normal_border,
        min_cluster_pixels,
        max_cluster_pixels);
}
