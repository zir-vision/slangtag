import "./image_buffer.slang";

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
    uint2 pixel = dispatch_thread_id.xy;
    if (!image_in_bounds(pc.input_size, pixel))
        return;

    let x_decimate = pc.input_size.width / pc.decimated_size.width;
    let y_decimate = pc.input_size.height / pc.decimated_size.height;

    if (x_decimate != y_decimate)
        return;

    let decimate_factor = x_decimate;
    if (pixel.x % decimate_factor == 0u && pixel.y % decimate_factor == 0u)
    {
        uint2 out_pixel = pixel / decimate_factor;
        if (image_in_bounds(pc.decimated_size, out_pixel))
        {
            image_u8_write(
                decimated_image,
                pc.decimated_size,
                out_pixel,
                image_u8_read(input_image, pc.input_size, pixel));
        }
    }
}
