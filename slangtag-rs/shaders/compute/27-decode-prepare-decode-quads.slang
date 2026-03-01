import "./image_buffer.slang";
import "../decode.slang";

StructuredBuffer<uint8_t> image_gray;
StructuredBuffer<uint32_t> corners;
RWStructuredBuffer<uint32_t> quad_params_out;

struct PushConstants
{
    ImageSize image_size;
    uint32_t quad_count;
    uint32_t marker_size_with_borders;
    uint32_t cell_size;
    float min_stddev_otsu;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    prepare_decode_quads_buffer(
        dispatch_thread_id.x,
        image_gray,
        pc.image_size,
        corners,
        quad_params_out,
        pc.quad_count,
        pc.marker_size_with_borders,
        pc.cell_size,
        pc.min_stddev_otsu);
}
