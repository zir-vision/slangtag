import "../select.slang";

[shader("compute")]
[numthreads(1, 1, 1)]
void count_nonzero_blob_diff_points(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    StructuredBuffer<uint32_t> input,
    RWStructuredBuffer<uint32_t> count_out,
    uint32_t total_points)
{
    count_nonzero_blob_diff_points(dispatch_thread_id.x, input, count_out, total_points);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void filter_nonzero_blob_diff_points(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    StructuredBuffer<uint32_t> input,
    RWStructuredBuffer<uint32_t> output,
    RWStructuredBuffer<uint32_t> output_count,
    uint32_t total_points)
{
    filter_nonzero_blob_diff_points(
        dispatch_thread_id.x,
        input,
        output,
        output_count,
        total_points);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void count_valid_peaks(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    StructuredBuffer<uint32_t> input,
    RWStructuredBuffer<uint32_t> count_out,
    uint32_t total_peaks)
{
    count_valid_peaks(dispatch_thread_id.x, input, count_out, total_peaks);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void filter_valid_peaks(
    uint3 dispatch_thread_id : SV_DispatchThreadID,
    StructuredBuffer<uint32_t> input,
    RWStructuredBuffer<uint32_t> output,
    RWStructuredBuffer<uint32_t> output_count,
    uint32_t total_peaks)
{
    filter_valid_peaks(dispatch_thread_id.x, input, output, output_count, total_peaks);
}
