import "./image_buffer.slang";
import "../threshold.slang";

StructuredBuffer<uint32_t> minmax_image;
RWStructuredBuffer<uint32_t> filtered_image;

struct PushConstants
{
    ImageSize minmax_size;
    ImageSize filtered_size;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void filter_minmax(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    filter_minmax_buffer(dispatch_thread_id.xy, minmax_image, pc.minmax_size, filtered_image, pc.filtered_size);
}
