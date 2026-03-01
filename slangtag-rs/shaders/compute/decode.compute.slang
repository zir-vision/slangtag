import "../decode.slang";

[shader("compute")]
[numthreads(1, 1, 1)]
void prepare_decode_quads(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    Texture2D<uint8_t1> image_gray,
    StructuredBuffer<uint32_t> corners,
    RWStructuredBuffer<uint32_t> quad_params_out,
    uint32_t quad_count,
    uint32_t marker_size_with_borders,
    uint32_t cell_size,
    float min_stddev_otsu)
{
    prepare_decode_quads(
        dispatch_thread_id.x,
        image_gray,
        corners,
        quad_params_out,
        quad_count,
        marker_size_with_borders,
        cell_size,
        min_stddev_otsu);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void extract_candidate_bits(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    Texture2D<uint8_t1> image_gray,
    StructuredBuffer<uint32_t> quad_params,
    RWStructuredBuffer<uint32_t> bits_out,
    uint32_t quad_count,
    uint32_t marker_size_with_borders,
    uint32_t cell_size,
    uint32_t cell_margin_pixels,
    uint32_t cell_span)
{
    extract_candidate_bits(
        dispatch_thread_id.x,
        image_gray,
        quad_params,
        bits_out,
        quad_count,
        marker_size_with_borders,
        cell_size,
        cell_margin_pixels,
        cell_span);
}
