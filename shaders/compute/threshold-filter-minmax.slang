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
    uint2 pixel = dispatch_thread_id.xy;
    if (!image_in_bounds(pc.minmax_size, pixel))
        return;

    uint8_t2 result = uint8_t2(255u, 0u);
    [ForceUnroll]
    for (int i = -1; i <= 1; ++i)
    {
        [ForceUnroll]
        for (int j = -1; j <= 1; ++j)
        {
            int nx = int(pixel.x) + i;
            int ny = int(pixel.y) + j;
            if (nx < 0 || ny < 0 || nx >= int(pc.minmax_size.width) || ny >= int(pc.minmax_size.height))
                continue;

            uint2 neighbor = uint2(uint32_t(nx), uint32_t(ny));
            result = minmax_minmax(result, image_rg8_read(minmax_image, pc.minmax_size, neighbor));
        }
    }

    image_rg8_write(filtered_image, pc.filtered_size, pixel, result);
}
