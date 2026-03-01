import "../sort.slang";

RWStructuredBuffer<uint32_t> points;
uint32_t total_points;
uint32_t j;
uint32_t k;

[shader("compute")]
[numthreads(1, 1, 1)]
void bitonic_sort_blob_diff_points(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    bitonic_sort_blob_diff_points(dispatch_thread_id.x, points, total_points, j, k);
}
