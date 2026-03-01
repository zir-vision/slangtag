import "../ccl.slang";

RWStructuredBuffer<uint32_t> labels;
RWStructuredBuffer<uint32_t> union_markers_size_device;
uint2 dim;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    final_labeling(dispatch_thread_id.xy, labels, union_markers_size_device, dim);
}
