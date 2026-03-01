import "../select.slang";

StructuredBuffer<uint32_t> input;
RWStructuredBuffer<uint32_t> count_out;
uint32_t total_points;

[shader("compute")]
[numthreads(1, 1, 1)]
void count_nonzero_blob_diff_points(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    count_nonzero_blob_diff_points(dispatch_thread_id.x, input, count_out, total_points);
}
