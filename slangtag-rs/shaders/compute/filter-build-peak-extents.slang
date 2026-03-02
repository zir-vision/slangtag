import "../filter.slang";

StructuredBuffer<uint32_t> sorted_peaks;
RWStructuredBuffer<uint32_t> peak_extents_out;
RWStructuredBuffer<uint32_t> peak_extent_count_out;
struct PushConstants
{
    uint32_t valid_peaks;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    build_peak_extents(
        dispatch_thread_id.x,
        sorted_peaks,
        peak_extents_out,
        peak_extent_count_out,
        pc.valid_peaks);
}
