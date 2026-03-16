import "./image_buffer.slang";
import "../ccl.slang";

StructuredBuffer<uint8_t> image;
RWStructuredBuffer<uint32_t> labels;

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
    init_buffer(dispatch_thread_id.xy, image, pc.image_size, labels);
}
