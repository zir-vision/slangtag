import "../filter.slang";

StructuredBuffer<uint32_t> sorted_peaks;
StructuredBuffer<uint32_t> peak_extents;
StructuredBuffer<uint32_t> line_fit_points;
StructuredBuffer<uint32_t> filtered_blob_extents;
RWStructuredBuffer<uint32_t> fitted_quads_out;
RWStructuredBuffer<uint32_t> fitted_quad_count_out;

struct PushConstants
{
    uint32_t peak_extent_count;
    uint32_t filtered_blob_extent_count;
    uint32_t max_nmaxima;
    float max_line_fit_mse;
    float cos_critical_rad;
    uint32_t min_tag_width;
    float quad_decimate;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    fit_quads(
        dispatch_thread_id.x,
        sorted_peaks,
        peak_extents,
        line_fit_points,
        filtered_blob_extents,
        fitted_quads_out,
        fitted_quad_count_out,
        pc.peak_extent_count,
        pc.filtered_blob_extent_count,
        pc.max_nmaxima,
        pc.max_line_fit_mse,
        pc.cos_critical_rad,
        pc.min_tag_width,
        pc.quad_decimate);
}
