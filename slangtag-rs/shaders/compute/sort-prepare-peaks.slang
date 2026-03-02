import "../sort.slang";

StructuredBuffer<uint32_t> input;
RWStructuredBuffer<uint32_t> output;
struct PushConstants
{
    uint32_t valid_peaks;
    uint32_t total_peaks;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    prepare_peaks(
        dispatch_thread_id.x,
        input,
        output,
        pc.valid_peaks,
        pc.total_peaks);
}
