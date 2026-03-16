import "./image_buffer.slang";

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
    uint2 pixel = dispatch_thread_id.xy;
    if (!image_in_bounds(pc.decimated_size, pixel))
        return;

    uint2 minmax_pixel = pixel / 4u;
    if (!image_in_bounds(pc.filtered_size, minmax_pixel))
        return;

    uint8_t2 minmax = image_rg8_read(filtered_image, pc.filtered_size, minmax_pixel);
    uint8_t result = 0u;

    if (minmax.y - minmax.x < pc.min_white_black_diff)
    {
        result = 127u;
    }
    else
    {
        uint threshold_value = uint(minmax.x) + (uint(minmax.y) - uint(minmax.x)) / 2u;
        result = image_u8_read(decimated_image, pc.decimated_size, pixel) > threshold_value ? 255u : 0u;
    }

    image_u8_write(thresholded_image, pc.thresholded_size, pixel, result);
}
