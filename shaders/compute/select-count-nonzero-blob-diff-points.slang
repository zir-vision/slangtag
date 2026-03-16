import "../select.slang";

StructuredBuffer<uint32_t> input;
RWStructuredBuffer<uint32_t> count_out;

struct PushConstants
{
    uint32_t total_points;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint tid = dispatch_thread_id.x;
    if (tid >= pc.total_points)
        return;
    if (!blob_diff_point_is_nonzero(input, tid))
        return;
    uint32_t old_count;
    InterlockedAdd(count_out[0], 1u, old_count);
}
