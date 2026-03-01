import "../ccl.slang";

[shader("compute")]
[numthreads(1, 1, 1)]
void init(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    RWTexture2D<uint8_t1> image,
    RWStructuredBuffer<uint32_t> labels)
{
    init(dispatch_thread_id.xy, image, labels);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void compression(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    RWStructuredBuffer<uint32_t> labels,
    uint2 dim)
{
    compression(dispatch_thread_id.xy, labels, dim);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void merge(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    RWStructuredBuffer<uint32_t> labels,
    uint2 dim)
{
    merge(dispatch_thread_id.xy, labels, dim);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void final_labeling(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    RWStructuredBuffer<uint32_t> labels,
    RWStructuredBuffer<uint32_t> union_markers_size_device,
    uint2 dim)
{
    final_labeling(dispatch_thread_id.xy, labels, union_markers_size_device, dim);
}
