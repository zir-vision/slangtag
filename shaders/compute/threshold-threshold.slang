import "./image_buffer.slang";
import "../threshold.slang";

StructuredBuffer<uint8_t> decimated_image;
StructuredBuffer<uint8_t2> filtered_image;
RWStructuredBuffer<uint8_t> thresholded_image;

struct PushConstants
{
    ImageSize decimated_size;
    ImageSize filtered_size;
    ImageSize thresholded_size;
    uint min_white_black_diff;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    threshold_buffer(
        dispatch_thread_id.xy,
        decimated_image,
        pc.decimated_size,
        filtered_image,
        pc.filtered_size,
        thresholded_image,
        pc.thresholded_size,
        pc.min_white_black_diff);
}
