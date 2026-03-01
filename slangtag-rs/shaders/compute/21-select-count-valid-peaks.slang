import "../select.slang";

StructuredBuffer<uint32_t> input;
RWStructuredBuffer<uint32_t> count_out;
uint32_t total_peaks;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    count_valid_peaks(dispatch_thread_id.x, input, count_out, total_peaks);
}
