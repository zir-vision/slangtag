import "./image_buffer.slang";
import "../threshold.slang";

StructuredBuffer<uint8_t> input_image;
RWStructuredBuffer<uint8_t2> minmax_image;

struct PushConstants
{
    ImageSize input_size;
    ImageSize minmax_size;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    minmax_buffer(dispatch_thread_id.xy, input_image, pc.input_size, minmax_image, pc.minmax_size);
}
