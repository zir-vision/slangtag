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
    count_nonzero_blob_diff_points(dispatch_thread_id.x, input, count_out, pc.total_points);
}
