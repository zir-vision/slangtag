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
[numthreads(16, 16, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint2 pixel = dispatch_thread_id.xy;
    if (!image_in_bounds(pc.minmax_size, pixel))
        return;

    uint8_t2 vals[4];
    for (int i = 0; i < 4; ++i)
    {
        uint2 row_base = (pixel * 4u) + uint2(0u, uint32_t(i));
        uint8_t4 row = uint8_t4(
            image_u8_read(input_image, pc.input_size, row_base + uint2(0u, 0u)),
            image_u8_read(input_image, pc.input_size, row_base + uint2(1u, 0u)),
            image_u8_read(input_image, pc.input_size, row_base + uint2(2u, 0u)),
            image_u8_read(input_image, pc.input_size, row_base + uint2(3u, 0u)));
        vals[i] = minmax_row(row);
    }

    image_rg8_write(
        minmax_image,
        pc.minmax_size,
        pixel,
        minmax_minmax(minmax_minmax(vals[0], vals[1]), minmax_minmax(vals[2], vals[3])));
}
