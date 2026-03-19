import "./image_buffer.slang";
import "../decode.slang";
import "./detect-control.slang";

StructuredBuffer<uint8_t> image_gray;
StructuredBuffer<uint32_t> fitted_quads;
RWStructuredBuffer<uint32_t> quad_params_out;
StructuredBuffer<uint32_t> control;

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
    uint32_t quad_count = control[kControlFittedQuadCountWord];
    if (tid >= quad_count)
        return;

    uint32_t quad_index = tid;
    uint32_t result_size = pc.marker_size_with_borders * pc.cell_size;

    float h00;
    float h01;
    float h02;
    float h10;
    float h11;
    float h12;
    float h20;
    float h21;
    float h22;
    bool valid_h = build_dst_to_src_homography_from_fitted_quads(
        fitted_quads, quad_index, result_size, h00, h01, h02, h10, h11, h12, h20, h21, h22);
    write_quad_param_float(quad_params_out, quad_index, kQuadParamH00Word, h00);
    write_quad_param_float(quad_params_out, quad_index, kQuadParamH01Word, h01);
    write_quad_param_float(quad_params_out, quad_index, kQuadParamH02Word, h02);
    write_quad_param_float(quad_params_out, quad_index, kQuadParamH10Word, h10);
    write_quad_param_float(quad_params_out, quad_index, kQuadParamH11Word, h11);
    write_quad_param_float(quad_params_out, quad_index, kQuadParamH12Word, h12);
    write_quad_param_float(quad_params_out, quad_index, kQuadParamH20Word, h20);
    write_quad_param_float(quad_params_out, quad_index, kQuadParamH21Word, h21);
    write_quad_param_float(quad_params_out, quad_index, kQuadParamH22Word, h22);
    if (!valid_h)
    {
        write_quad_param_uint(quad_params_out, quad_index, kQuadParamModeWord, 2u);
        write_quad_param_uint(quad_params_out, quad_index, kQuadParamFillBitWord, 0u);
        write_quad_param_uint(quad_params_out, quad_index, kQuadParamOtsuThresholdWord, 0u);
        return;
    }

    uint32_t inner_margin = pc.cell_size / 2u;
    uint32_t inner_start = inner_margin;
    uint32_t inner_end = result_size - inner_margin;
    uint32_t inner_count = 0u;
    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (uint32_t y = inner_start; y < inner_end; ++y)
    {
        for (uint32_t x = inner_start; x < inner_end; ++x)
        {
            float v = float(sample_warped_buffer(image_gray, pc.image_size, h00, h01, h02, h10, h11, h12, h20, h21, h22, x, y));
            sum += v;
            sum_sq += v * v;
            inner_count += 1u;
        }
    }

    float mean = inner_count > 0u ? sum / float(inner_count) : 0.0f;
    float variance = inner_count > 0u ? max(0.0f, sum_sq / float(inner_count) - mean * mean) : 0.0f;
    float stddev = sqrt(variance);

    if (stddev < pc.min_stddev_otsu)
    {
        uint32_t fill = mean > 127.0f ? 1u : 0u;
        write_quad_param_uint(quad_params_out, quad_index, kQuadParamModeWord, 1u);
        write_quad_param_uint(quad_params_out, quad_index, kQuadParamFillBitWord, fill);
        write_quad_param_uint(quad_params_out, quad_index, kQuadParamOtsuThresholdWord, 0u);
        return;
    }

    uint32_t hist[256];
    [ForceUnroll]
    for (uint32_t i = 0u; i < 256u; ++i)
    {
        hist[i] = 0u;
    }

    uint32_t total = result_size * result_size;
    float total_sum = 0.0f;

    for (uint32_t y = 0u; y < result_size; ++y)
    {
        for (uint32_t x = 0u; x < result_size; ++x)
        {
            uint32_t v = uint32_t(sample_warped_buffer(image_gray, pc.image_size, h00, h01, h02, h10, h11, h12, h20, h21, h22, x, y));
            hist[v] += 1u;
            total_sum += float(v);
        }
    }

    uint32_t otsu_threshold = 0u;
    float sum_b = 0.0f;
    uint32_t w_b = 0u;
    float max_between = -1.0f;

    for (uint32_t t = 0u; t < 256u; ++t)
    {
        w_b += hist[t];
        if (w_b == 0u)
            continue;

        uint32_t w_f = total - w_b;
        if (w_f == 0u)
            break;

        sum_b += float(t) * float(hist[t]);
        float m_b = sum_b / float(w_b);
        float m_f = (total_sum - sum_b) / float(w_f);
        float between = float(w_b) * float(w_f) * (m_b - m_f) * (m_b - m_f);
        if (between > max_between)
        {
            max_between = between;
            otsu_threshold = t;
        }
    }
    write_quad_param_uint(quad_params_out, quad_index, kQuadParamModeWord, 0u);
    write_quad_param_uint(quad_params_out, quad_index, kQuadParamFillBitWord, 0u);
    write_quad_param_uint(quad_params_out, quad_index, kQuadParamOtsuThresholdWord, otsu_threshold);
}
