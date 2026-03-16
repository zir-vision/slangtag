import "./image_buffer.slang";
import "../ccl.slang";

RWStructuredBuffer<uint32_t> labels;
RWStructuredBuffer<uint32_t> union_markers_size_device;

struct PushConstants
{
    ImageSize image_size;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint2 dim = uint2(pc.image_size.width, pc.image_size.height);
    final_labeling(dispatch_thread_id.xy, labels, union_markers_size_device, dim);
}
