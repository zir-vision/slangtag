import "../filter.slang";

StructuredBuffer<uint32_t> sorted_peaks;
StructuredBuffer<uint32_t> peak_extents;
StructuredBuffer<uint32_t> line_fit_points;
StructuredBuffer<uint32_t> filtered_blob_extents;
RWStructuredBuffer<uint32_t> fitted_quads_out;
RWStructuredBuffer<uint32_t> fitted_quad_count_out;
uint32_t peak_extent_count;
uint32_t filtered_blob_extent_count;
uint32_t max_nmaxima;
float max_line_fit_mse;
float cos_critical_rad;
uint32_t min_tag_width;
float quad_decimate;

[shader("compute")]
[numthreads(1, 1, 1)]
void fit_quads(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    fit_quads(
        dispatch_thread_id.x,
        sorted_peaks,
        peak_extents,
        line_fit_points,
        filtered_blob_extents,
        fitted_quads_out,
        fitted_quad_count_out,
        peak_extent_count,
        filtered_blob_extent_count,
        max_nmaxima,
        max_line_fit_mse,
        cos_critical_rad,
        min_tag_width,
        quad_decimate);
}
