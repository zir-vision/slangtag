import "./image_buffer.slang";
import "../decode.slang";

StructuredBuffer<uint32_t> image_gray;
StructuredBuffer<uint32_t> quad_params;
RWStructuredBuffer<uint32_t> bits_out;

struct PushConstants
{
    ImageSize image_size;
    uint32_t quad_count;
    uint32_t marker_size_with_borders;
    uint32_t cell_size;
    uint32_t cell_margin_pixels;
    uint32_t cell_span;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void extract_candidate_bits(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    extract_candidate_bits_buffer(
        dispatch_thread_id.x,
        image_gray,
        pc.image_size,
        quad_params,
        bits_out,
        pc.quad_count,
        pc.marker_size_with_borders,
        pc.cell_size,
        pc.cell_margin_pixels,
        pc.cell_span);
}
