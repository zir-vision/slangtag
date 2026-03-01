import "../select.slang";

StructuredBuffer<uint32_t> input;
RWStructuredBuffer<uint32_t> output;
RWStructuredBuffer<uint32_t> output_count;
uint32_t total_peaks;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    filter_valid_peaks(dispatch_thread_id.x, input, output, output_count, total_peaks);
}
