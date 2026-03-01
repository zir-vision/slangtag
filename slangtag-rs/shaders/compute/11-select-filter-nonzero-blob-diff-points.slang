import "../select.slang";

StructuredBuffer<uint32_t> input;
RWStructuredBuffer<uint32_t> output;
RWStructuredBuffer<uint32_t> output_count;
uint32_t total_points;

[shader("compute")]
[numthreads(1, 1, 1)]
void filter_nonzero_blob_diff_points(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    filter_nonzero_blob_diff_points(
        dispatch_thread_id.x,
        input,
        output,
        output_count,
        total_points);
}
