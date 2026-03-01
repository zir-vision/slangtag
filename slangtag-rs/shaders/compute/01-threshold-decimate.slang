import "./image_buffer.slang";
import "../threshold.slang";

StructuredBuffer<uint32_t> input_image;
RWStructuredBuffer<uint32_t> decimated_image;

struct PushConstants
{
    ImageSize input_size;
    ImageSize decimated_size;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void decimate(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    decimate_buffer(dispatch_thread_id.xy, input_image, pc.input_size, decimated_image, pc.decimated_size);
}
