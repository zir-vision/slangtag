import "../filter.slang";

[shader("compute")]
[numthreads(1, 1, 1)]
void build_blob_pair_extents(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    StructuredBuffer<uint32_t> sorted_points,
    RWStructuredBuffer<uint32_t> extents_out,
    RWStructuredBuffer<uint32_t> extent_count_out,
    uint32_t valid_points)
{
    build_blob_pair_extents(
        dispatch_thread_id.x,
        sorted_points,
        extents_out,
        extent_count_out,
        valid_points);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void filter_blob_pair_extents(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    StructuredBuffer<uint32_t> extents_in,
    RWStructuredBuffer<uint32_t> extents_out,
    RWStructuredBuffer<uint32_t> selected_extent_count_out,
    RWStructuredBuffer<uint32_t> selected_point_count_out,
    uint32_t extent_count,
    uint32_t tag_width,
    uint32_t reversed_border,
    uint32_t normal_border,
    uint32_t min_cluster_pixels,
    uint32_t max_cluster_pixels)
{
    filter_blob_pair_extents(
        dispatch_thread_id.x,
        extents_in,
        extents_out,
        selected_extent_count_out,
        selected_point_count_out,
        extent_count,
        tag_width,
        reversed_border,
        normal_border,
        min_cluster_pixels,
        max_cluster_pixels);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void rewrite_selected_blob_points_with_theta(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    StructuredBuffer<uint32_t> sorted_points,
    StructuredBuffer<uint32_t> extents_in,
    StructuredBuffer<uint32_t> filtered_extents,
    RWStructuredBuffer<uint32_t> selected_points_out,
    uint32_t extent_count,
    uint32_t valid_points)
{
    rewrite_selected_blob_points_with_theta(
        dispatch_thread_id.x,
        sorted_points,
        extents_in,
        filtered_extents,
        selected_points_out,
        extent_count,
        valid_points);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void build_line_fit_points(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    StructuredBuffer<uint32_t> sorted_selected_points,
    Texture2D<uint8_t1> decimated_image,
    RWStructuredBuffer<uint32_t> line_fit_points_out,
    uint32_t point_count,
    uint32_t decimated_width,
    uint32_t decimated_height,
    uint32_t decimate)
{
    build_line_fit_points(
        dispatch_thread_id.x,
        sorted_selected_points,
        decimated_image,
        line_fit_points_out,
        point_count,
        decimated_width,
        decimated_height,
        decimate);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void fit_line_errors_and_peaks(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    StructuredBuffer<uint32_t> line_fit_points,
    StructuredBuffer<uint32_t> filtered_extents,
    RWStructuredBuffer<uint32_t> errs_out,
    RWStructuredBuffer<uint32_t> filtered_errs_out,
    RWStructuredBuffer<uint32_t> peaks_out,
    uint32_t extent_count,
    uint32_t point_count)
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

[shader("compute")]
[numthreads(1, 1, 1)]
void build_peak_extents(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    StructuredBuffer<uint32_t> sorted_peaks,
    RWStructuredBuffer<uint32_t> peak_extents_out,
    RWStructuredBuffer<uint32_t> peak_extent_count_out,
    uint32_t valid_peaks)
{
    build_peak_extents(
        dispatch_thread_id.x,
        sorted_peaks,
        peak_extents_out,
        peak_extent_count_out,
        valid_peaks);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void fit_quads(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    StructuredBuffer<uint32_t> sorted_peaks,
    StructuredBuffer<uint32_t> peak_extents,
    StructuredBuffer<uint32_t> line_fit_points,
    StructuredBuffer<uint32_t> filtered_blob_extents,
    RWStructuredBuffer<uint32_t> fitted_quads_out,
    RWStructuredBuffer<uint32_t> fitted_quad_count_out,
    uint32_t peak_extent_count,
    uint32_t filtered_blob_extent_count,
    uint32_t max_nmaxima,
    float max_line_fit_mse,
    float cos_critical_rad,
    uint32_t min_tag_width,
    float quad_decimate)
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
