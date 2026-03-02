import "../filter.slang";

StructuredBuffer<uint32_t> sorted_points;
RWStructuredBuffer<uint32_t> extents_out;
RWStructuredBuffer<uint32_t> extent_count_out;
struct PushConstants
{
    uint32_t valid_points;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    build_blob_pair_extents(
        dispatch_thread_id.x,
        sorted_points,
        extents_out,
        extent_count_out,
        pc.valid_points);
}
