import "./image_buffer.slang";
import "../threshold.slang";

StructuredBuffer<uint8_t2> minmax_image;
RWStructuredBuffer<uint8_t2> filtered_image;

struct PushConstants
{
    ImageSize minmax_size;
    ImageSize filtered_size;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    filter_minmax_buffer(dispatch_thread_id.xy, minmax_image, pc.minmax_size, filtered_image, pc.filtered_size);
}
