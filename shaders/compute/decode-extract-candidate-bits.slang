import "./image_buffer.slang";
import "../decode.slang";

StructuredBuffer<uint8_t> image_gray;
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

uint8_t sample_image_buffer(StructuredBuffer<uint8_t> image, ImageSize size, int x, int y)
{
    return image_u8_read_i32(image, size, x, y);
}

uint8_t sample_warped_buffer(StructuredBuffer<uint8_t> image,
                             ImageSize size,
                             float h00,
                             float h01,
                             float h02,
                             float h10,
                             float h11,
                             float h12,
                             float h20,
                             float h21,
                             float h22,
                             uint32_t dx,
                             uint32_t dy)
{
    float fx = float(dx);
    float fy = float(dy);
    float denom = h20 * fx + h21 * fy + h22;
    if (abs(denom) <= 1e-8f)
        return 0u;

    float sx = (h00 * fx + h01 * fy + h02) / denom;
    float sy = (h10 * fx + h11 * fy + h12) / denom;
    int src_x = int(floor(sx + 0.5f));
    int src_y = int(floor(sy + 0.5f));
    return sample_image_buffer(image, size, src_x, src_y);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint tid = dispatch_thread_id.x;
    uint32_t cells_per_quad = pc.marker_size_with_borders * pc.marker_size_with_borders;
    uint32_t total_cells = pc.quad_count * cells_per_quad;
    if (tid >= total_cells || pc.cell_span == 0u)
        return;

    uint32_t quad_index = tid / cells_per_quad;
    uint32_t cell_index = tid - quad_index * cells_per_quad;
    uint32_t cell_y = cell_index / pc.marker_size_with_borders;
    uint32_t cell_x = cell_index - cell_y * pc.marker_size_with_borders;

    uint32_t mode = read_quad_param_uint(quad_params, quad_index, kQuadParamModeWord);
    if (mode == 2u)
    {
        bits_out[tid] = 0u;
        return;
    }
    if (mode == 1u)
    {
        bits_out[tid] = read_quad_param_uint(quad_params, quad_index, kQuadParamFillBitWord);
        return;
    }

    float h00 = read_quad_param_float(quad_params, quad_index, kQuadParamH00Word);
    float h01 = read_quad_param_float(quad_params, quad_index, kQuadParamH01Word);
    float h02 = read_quad_param_float(quad_params, quad_index, kQuadParamH02Word);
    float h10 = read_quad_param_float(quad_params, quad_index, kQuadParamH10Word);
    float h11 = read_quad_param_float(quad_params, quad_index, kQuadParamH11Word);
    float h12 = read_quad_param_float(quad_params, quad_index, kQuadParamH12Word);
    float h20 = read_quad_param_float(quad_params, quad_index, kQuadParamH20Word);
    float h21 = read_quad_param_float(quad_params, quad_index, kQuadParamH21Word);
    float h22 = read_quad_param_float(quad_params, quad_index, kQuadParamH22Word);
    uint32_t otsu_threshold = read_quad_param_uint(quad_params, quad_index, kQuadParamOtsuThresholdWord);

    uint32_t area = pc.cell_span * pc.cell_span;
    uint32_t start_x = cell_x * pc.cell_size + pc.cell_margin_pixels;
    uint32_t start_y = cell_y * pc.cell_size + pc.cell_margin_pixels;
    uint32_t white_pixels = 0u;

    for (uint32_t yy = 0u; yy < pc.cell_span; ++yy)
    {
        for (uint32_t xx = 0u; xx < pc.cell_span; ++xx)
        {
            uint32_t v = uint32_t(sample_warped_buffer(
                image_gray,
                pc.image_size,
                h00,
                h01,
                h02,
                h10,
                h11,
                h12,
                h20,
                h21,
                h22,
                start_x + xx,
                start_y + yy));
            if (v > otsu_threshold)
                white_pixels += 1u;
        }
    }
    bits_out[tid] = (white_pixels > (area >> 1u)) ? 1u : 0u;
}
