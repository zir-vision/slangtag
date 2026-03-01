import "./image_buffer.slang";
import "../ccl.slang";

StructuredBuffer<uint32_t> image;
RWStructuredBuffer<uint32_t> labels;

struct PushConstants
{
    ImageSize image_size;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void init(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    init_buffer(dispatch_thread_id.xy, image, pc.image_size, labels);
}
