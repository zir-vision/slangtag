import "../threshold.slang";

[shader("compute")]
[numthreads(1, 1, 1)]
void decimate(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    Texture2D<uint8_t1> input_texture,
    RWTexture2D<uint8_t1> decimated_texture)
{
    decimate(dispatch_thread_id.xy, input_texture, decimated_texture);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void minmax(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    Texture2D<uint8_t1> input_texture,
    RWTexture2D<uint8_t2> minmax_texture)
{
    minmax(dispatch_thread_id.xy, input_texture, minmax_texture);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void filter_minmax(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    Texture2D<uint8_t2> minmax_texture,
    RWTexture2D<uint8_t2> filtered_texture)
{
    filter_minmax(dispatch_thread_id.xy, minmax_texture, filtered_texture);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void threshold(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    Texture2D<uint8_t1> decimated_texture,
    Texture2D<uint8_t2> filtered_texture,
    RWTexture2D<uint8_t1> thresholded_texture,
    uint min_white_black_diff)
{
    threshold(
        dispatch_thread_id.xy,
        decimated_texture,
        filtered_texture,
        thresholded_texture,
        min_white_black_diff);
}
