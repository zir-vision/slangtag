import "../select.slang";

StructuredBuffer<uint32_t> input;
RWStructuredBuffer<uint32_t> output;
RWStructuredBuffer<uint32_t> output_count;
struct PushConstants
{
    uint32_t total_peaks;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    filter_valid_peaks(dispatch_thread_id.x, input, output, output_count, pc.total_peaks);
}
