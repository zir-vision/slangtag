import "../sort.slang";

RWStructuredBuffer<uint32_t> peaks;
uint32_t total_peaks;
uint32_t j;
uint32_t k;

[shader("compute")]
[numthreads(1, 1, 1)]
void bitonic_sort_peaks(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    bitonic_sort_peaks(dispatch_thread_id.x, peaks, total_peaks, j, k);
}
