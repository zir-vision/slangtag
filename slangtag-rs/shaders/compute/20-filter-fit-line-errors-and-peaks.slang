import "../filter.slang";

StructuredBuffer<uint32_t> line_fit_points;
StructuredBuffer<uint32_t> filtered_extents;
RWStructuredBuffer<uint32_t> errs_out;
RWStructuredBuffer<uint32_t> filtered_errs_out;
RWStructuredBuffer<uint32_t> peaks_out;
uint32_t extent_count;
uint32_t point_count;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    fit_line_errors_and_peaks(
        dispatch_thread_id.x,
        line_fit_points,
        filtered_extents,
        errs_out,
        filtered_errs_out,
        peaks_out,
        extent_count,
        point_count);
}
