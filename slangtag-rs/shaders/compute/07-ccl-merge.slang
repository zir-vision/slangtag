import "./image_buffer.slang";
import "../ccl.slang";

RWStructuredBuffer<uint32_t> labels;

struct PushConstants
{
    ImageSize image_size;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint2 dim = uint2(pc.image_size.width, pc.image_size.height);
    merge(dispatch_thread_id.xy, labels, dim);
}
