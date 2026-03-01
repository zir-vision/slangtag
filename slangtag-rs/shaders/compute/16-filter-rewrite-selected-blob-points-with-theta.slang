import "../filter.slang";

StructuredBuffer<uint32_t> sorted_points;
StructuredBuffer<uint32_t> extents_in;
StructuredBuffer<uint32_t> filtered_extents;
RWStructuredBuffer<uint32_t> selected_points_out;
uint32_t extent_count;
uint32_t valid_points;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    rewrite_selected_blob_points_with_theta(
        dispatch_thread_id.x,
        sorted_points,
        extents_in,
        filtered_extents,
        selected_points_out,
        extent_count,
        valid_points);
}
