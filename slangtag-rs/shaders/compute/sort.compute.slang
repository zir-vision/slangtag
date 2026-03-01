import "../sort.slang";

[shader("compute")]
[numthreads(1, 1, 1)]
void prepare_blob_diff_points(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    StructuredBuffer<uint32_t> input,
    RWStructuredBuffer<uint32_t> output,
    uint32_t valid_points,
    uint32_t total_points)
{
    prepare_blob_diff_points(dispatch_thread_id.x, input, output, valid_points, total_points);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void bitonic_sort_blob_diff_points(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    RWStructuredBuffer<uint32_t> points,
    uint32_t total_points,
    uint32_t j,
    uint32_t k)
{
    bitonic_sort_blob_diff_points(dispatch_thread_id.x, points, total_points, j, k);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void prepare_selected_blob_points(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    StructuredBuffer<uint32_t> input,
    RWStructuredBuffer<uint32_t> output,
    uint32_t valid_points,
    uint32_t total_points)
{
    prepare_selected_blob_points(dispatch_thread_id.x, input, output, valid_points, total_points);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void bitonic_sort_selected_blob_points(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    RWStructuredBuffer<uint32_t> points,
    uint32_t total_points,
    uint32_t j,
    uint32_t k)
{
    bitonic_sort_selected_blob_points(dispatch_thread_id.x, points, total_points, j, k);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void prepare_peaks(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    StructuredBuffer<uint32_t> input,
    RWStructuredBuffer<uint32_t> output,
    uint32_t valid_peaks,
    uint32_t total_peaks)
{
    prepare_peaks(dispatch_thread_id.x, input, output, valid_peaks, total_peaks);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void bitonic_sort_peaks(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    RWStructuredBuffer<uint32_t> peaks,
    uint32_t total_peaks,
    uint32_t j,
    uint32_t k)
{
    bitonic_sort_peaks(dispatch_thread_id.x, peaks, total_peaks, j, k);
}
