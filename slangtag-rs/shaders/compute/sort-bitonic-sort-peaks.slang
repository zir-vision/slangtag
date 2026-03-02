import "../sort.slang";

RWStructuredBuffer<uint32_t> peaks;
struct PushConstants
{
    uint32_t total_peaks;
    uint32_t j;
    uint32_t k;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    bitonic_sort_peaks(
        dispatch_thread_id.x,
        peaks,
        pc.total_peaks,
        pc.j,
        pc.k);
}
