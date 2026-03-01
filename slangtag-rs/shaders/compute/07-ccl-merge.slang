import "../ccl.slang";

RWStructuredBuffer<uint32_t> labels;
uint2 dim;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    merge(dispatch_thread_id.xy, labels, dim);
}
