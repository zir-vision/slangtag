import "../filter.slang";

StructuredBuffer<uint32_t> sorted_peaks;
RWStructuredBuffer<uint32_t> peak_extents_out;
RWStructuredBuffer<uint32_t> peak_extent_count_out;
uint32_t valid_peaks;

[shader("compute")]
[numthreads(1, 1, 1)]
void build_peak_extents(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    build_peak_extents(
        dispatch_thread_id.x,
        sorted_peaks,
        peak_extents_out,
        peak_extent_count_out,
        valid_peaks);
}
