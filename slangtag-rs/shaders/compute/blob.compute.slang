import "../blob.slang";

[shader("compute")]
[numthreads(1, 1, 1)]
void blob_diff(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    Texture2D<uint8_t1> thresholded_image,
    RWStructuredBuffer<uint32_t> blobs,
    RWStructuredBuffer<uint32_t> union_markers_size,
    RWStructuredBuffer<uint32_t> result,
    uint2 dim,
    uint32_t min_blob_size)
{
    blob_diff(
        dispatch_thread_id.xy,
        thresholded_image,
        blobs,
        union_markers_size,
        result,
        dim,
        min_blob_size);
}
