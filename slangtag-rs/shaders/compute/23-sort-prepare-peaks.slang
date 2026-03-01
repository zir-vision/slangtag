import "../sort.slang";

StructuredBuffer<uint32_t> input;
RWStructuredBuffer<uint32_t> output;
uint32_t valid_peaks;
uint32_t total_peaks;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    prepare_peaks(dispatch_thread_id.x, input, output, valid_peaks, total_peaks);
}
