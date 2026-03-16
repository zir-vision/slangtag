import "./image_buffer.slang";
import "../threshold.slang";

StructuredBuffer<uint8_t> input_image;
RWStructuredBuffer<uint8_t> decimated_image;

struct PushConstants
{
    ImageSize input_size;
    ImageSize decimated_size;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    decimate_buffer(dispatch_thread_id.xy, input_image, pc.input_size, decimated_image, pc.decimated_size);
}
