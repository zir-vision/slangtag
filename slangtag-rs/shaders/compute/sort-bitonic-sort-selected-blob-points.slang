import "../sort.slang";

RWStructuredBuffer<uint32_t> points;
struct PushConstants
{
    uint32_t total_points;
    uint32_t j;
    uint32_t k;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    bitonic_sort_selected_blob_points(
        dispatch_thread_id.x,
        points,
        pc.total_points,
        pc.j,
        pc.k);
}
