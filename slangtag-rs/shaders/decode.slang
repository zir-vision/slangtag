import "compute/image_buffer.slang";

static const uint32_t kCornersWordsPerQuad = 8u;
static const uint32_t kQuadParamWordsPerQuad = 12u;
static const uint32_t kQuadParamH00Word = 0u;
static const uint32_t kQuadParamH01Word = 1u;
static const uint32_t kQuadParamH02Word = 2u;
static const uint32_t kQuadParamH10Word = 3u;
static const uint32_t kQuadParamH11Word = 4u;
static const uint32_t kQuadParamH12Word = 5u;
static const uint32_t kQuadParamH20Word = 6u;
static const uint32_t kQuadParamH21Word = 7u;
static const uint32_t kQuadParamH22Word = 8u;
static const uint32_t kQuadParamOtsuThresholdWord = 9u;
static const uint32_t kQuadParamModeWord = 10u;
static const uint32_t kQuadParamFillBitWord = 11u;

float read_corner(StructuredBuffer<uint32_t> corners, uint32_t quad_index, uint32_t word_index)
{
    return asfloat(corners[quad_index * kCornersWordsPerQuad + word_index]);
}

bool solve_linear_8x8(inout float a[8][9], out float x[8])
{
    [loop]
    for (uint32_t col = 0u; col < 8u; ++col)
    {
        uint32_t pivot = col;
        float best = abs(a[pivot][col]);
        [loop]
        for (uint32_t row = col + 1u; row < 8u; ++row)
        {
            float v = abs(a[row][col]);
            if (v > best)
            {
                best = v;
                pivot = row;
            }
        }

        if (best <= 1e-8f)
            return false;

        if (pivot != col)
        {
            [loop]
            for (uint32_t k = col; k < 9u; ++k)
            {
                float tmp = a[col][k];
                a[col][k] = a[pivot][k];
                a[pivot][k] = tmp;
            }
        }

        float inv = 1.0f / a[col][col];
        [loop]
        for (uint32_t k = col; k < 9u; ++k)
        {
            a[col][k] *= inv;
        }

        [loop]
        for (uint32_t row = 0u; row < 8u; ++row)
        {
            if (row == col)
                continue;
            float factor = a[row][col];
            if (abs(factor) <= 1e-12f)
                continue;
            [loop]
            for (uint32_t k = col; k < 9u; ++k)
            {
                a[row][k] -= factor * a[col][k];
            }
        }
    }

    [unroll]
    for (uint32_t i = 0u; i < 8u; ++i)
    {
        x[i] = a[i][8];
    }
    return true;
}

bool build_dst_to_src_homography(StructuredBuffer<uint32_t> corners,
                                 uint32_t quad_index,
                                 uint32_t result_size,
                                 out float h00,
                                 out float h01,
                                 out float h02,
                                 out float h10,
                                 out float h11,
                                 out float h12,
                                 out float h20,
                                 out float h21,
                                 out float h22)
{
    float x0 = 0.0f;
    float y0 = 0.0f;
    float x1 = float(result_size - 1u);
    float y1 = 0.0f;
    float x2 = float(result_size - 1u);
    float y2 = float(result_size - 1u);
    float x3 = 0.0f;
    float y3 = float(result_size - 1u);

    float u0 = read_corner(corners, quad_index, 0u);
    float v0 = read_corner(corners, quad_index, 1u);
    float u1 = read_corner(corners, quad_index, 2u);
    float v1 = read_corner(corners, quad_index, 3u);
    float u2 = read_corner(corners, quad_index, 4u);
    float v2 = read_corner(corners, quad_index, 5u);
    float u3 = read_corner(corners, quad_index, 6u);
    float v3 = read_corner(corners, quad_index, 7u);

    float a[8][9];
    [unroll]
    for (uint32_t i = 0u; i < 8u; ++i)
    {
        [unroll]
        for (uint32_t j = 0u; j < 9u; ++j)
        {
            a[i][j] = 0.0f;
        }
    }

    // (x0,y0)->(u0,v0)
    a[0][0] = x0; a[0][1] = y0; a[0][2] = 1.0f; a[0][6] = -u0 * x0; a[0][7] = -u0 * y0; a[0][8] = u0;
    a[1][3] = x0; a[1][4] = y0; a[1][5] = 1.0f; a[1][6] = -v0 * x0; a[1][7] = -v0 * y0; a[1][8] = v0;
    // (x1,y1)->(u1,v1)
    a[2][0] = x1; a[2][1] = y1; a[2][2] = 1.0f; a[2][6] = -u1 * x1; a[2][7] = -u1 * y1; a[2][8] = u1;
    a[3][3] = x1; a[3][4] = y1; a[3][5] = 1.0f; a[3][6] = -v1 * x1; a[3][7] = -v1 * y1; a[3][8] = v1;
    // (x2,y2)->(u2,v2)
    a[4][0] = x2; a[4][1] = y2; a[4][2] = 1.0f; a[4][6] = -u2 * x2; a[4][7] = -u2 * y2; a[4][8] = u2;
    a[5][3] = x2; a[5][4] = y2; a[5][5] = 1.0f; a[5][6] = -v2 * x2; a[5][7] = -v2 * y2; a[5][8] = v2;
    // (x3,y3)->(u3,v3)
    a[6][0] = x3; a[6][1] = y3; a[6][2] = 1.0f; a[6][6] = -u3 * x3; a[6][7] = -u3 * y3; a[6][8] = u3;
    a[7][3] = x3; a[7][4] = y3; a[7][5] = 1.0f; a[7][6] = -v3 * x3; a[7][7] = -v3 * y3; a[7][8] = v3;

    float x[8];
    if (!solve_linear_8x8(a, x))
    {
        h00 = 1.0f; h01 = 0.0f; h02 = 0.0f;
        h10 = 0.0f; h11 = 1.0f; h12 = 0.0f;
        h20 = 0.0f; h21 = 0.0f; h22 = 1.0f;
        return false;
    }

    h00 = x[0];
    h01 = x[1];
    h02 = x[2];
    h10 = x[3];
    h11 = x[4];
    h12 = x[5];
    h20 = x[6];
    h21 = x[7];
    h22 = 1.0f;
    return true;
}

uint8_t sample_image(Texture2D<uint8_t1> image_gray, int x, int y)
{
    uint2 dim;
    image_gray.GetDimensions(dim.x, dim.y);
    if (dim.x == 0u || dim.y == 0u || x < 0 || y < 0 || x >= int(dim.x) || y >= int(dim.y))
        return 0u;
    return image_gray[uint2(uint32_t(x), uint32_t(y))].x;
}

uint8_t sample_warped(Texture2D<uint8_t1> image_gray,
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
    return sample_image(image_gray, src_x, src_y);
}

uint8_t sample_image_buffer(StructuredBuffer<uint8_t> image_gray, ImageSize image_size, int x, int y)
{
    return image_u8_read_i32(image_gray, image_size, x, y);
}

uint8_t sample_warped_buffer(StructuredBuffer<uint8_t> image_gray,
                             ImageSize image_size,
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
    return sample_image_buffer(image_gray, image_size, src_x, src_y);
}

void write_quad_param_float(RWStructuredBuffer<uint32_t> quad_params, uint32_t quad_index, uint32_t word_index, float value)
{
    quad_params[quad_index * kQuadParamWordsPerQuad + word_index] = asuint(value);
}

void write_quad_param_uint(RWStructuredBuffer<uint32_t> quad_params, uint32_t quad_index, uint32_t word_index, uint32_t value)
{
    quad_params[quad_index * kQuadParamWordsPerQuad + word_index] = value;
}

float read_quad_param_float(StructuredBuffer<uint32_t> quad_params, uint32_t quad_index, uint32_t word_index)
{
    return asfloat(quad_params[quad_index * kQuadParamWordsPerQuad + word_index]);
}

uint32_t read_quad_param_uint(StructuredBuffer<uint32_t> quad_params, uint32_t quad_index, uint32_t word_index)
{
    return quad_params[quad_index * kQuadParamWordsPerQuad + word_index];
}

void prepare_decode_quads(
    uint tid,
    Texture2D<uint8_t1> image_gray,
    StructuredBuffer<uint32_t> corners,
    RWStructuredBuffer<uint32_t> quad_params_out,
    uint32_t quad_count,
    uint32_t marker_size_with_borders,
    uint32_t cell_size,
    float min_stddev_otsu)
{
    if (tid >= quad_count)
        return;

    uint32_t quad_index = tid;
    uint32_t result_size = marker_size_with_borders * cell_size;

    float h00;
    float h01;
    float h02;
    float h10;
    float h11;
    float h12;
    float h20;
    float h21;
    float h22;
    bool valid_h = build_dst_to_src_homography(
        corners, quad_index, result_size, h00, h01, h02, h10, h11, h12, h20, h21, h22);
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

    uint32_t inner_margin = cell_size / 2u;
    uint32_t inner_start = inner_margin;
    uint32_t inner_end = result_size - inner_margin;
    uint32_t inner_count = 0u;
    float sum = 0.0f;
    float sum_sq = 0.0f;

    [loop]
    for (uint32_t y = inner_start; y < inner_end; ++y)
    {
        [loop]
        for (uint32_t x = inner_start; x < inner_end; ++x)
        {
            float v = float(sample_warped(image_gray, h00, h01, h02, h10, h11, h12, h20, h21, h22, x, y));
            sum += v;
            sum_sq += v * v;
            inner_count += 1u;
        }
    }

    float mean = inner_count > 0u ? sum / float(inner_count) : 0.0f;
    float variance = inner_count > 0u ? max(0.0f, sum_sq / float(inner_count) - mean * mean) : 0.0f;
    float stddev = sqrt(variance);

    if (stddev < min_stddev_otsu)
    {
        uint32_t fill = mean > 127.0f ? 1u : 0u;
        write_quad_param_uint(quad_params_out, quad_index, kQuadParamModeWord, 1u);
        write_quad_param_uint(quad_params_out, quad_index, kQuadParamFillBitWord, fill);
        write_quad_param_uint(quad_params_out, quad_index, kQuadParamOtsuThresholdWord, 0u);
        return;
    }

    uint32_t hist[256];
    [unroll]
    for (uint32_t i = 0u; i < 256u; ++i)
    {
        hist[i] = 0u;
    }

    uint32_t total = result_size * result_size;
    float total_sum = 0.0f;
    [loop]
    for (uint32_t y = 0u; y < result_size; ++y)
    {
        [loop]
        for (uint32_t x = 0u; x < result_size; ++x)
        {
            uint32_t v = uint32_t(sample_warped(image_gray, h00, h01, h02, h10, h11, h12, h20, h21, h22, x, y));
            hist[v] += 1u;
            total_sum += float(v);
        }
    }

    uint32_t otsu_threshold = 0u;
    float sum_b = 0.0f;
    uint32_t w_b = 0u;
    float max_between = -1.0f;
    [loop]
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

void extract_candidate_bits(
    uint tid,
    Texture2D<uint8_t1> image_gray,
    StructuredBuffer<uint32_t> quad_params,
    RWStructuredBuffer<uint32_t> bits_out,
    uint32_t quad_count,
    uint32_t marker_size_with_borders,
    uint32_t cell_size,
    uint32_t cell_margin_pixels,
    uint32_t cell_span)
{
    uint32_t cells_per_quad = marker_size_with_borders * marker_size_with_borders;
    uint32_t total_cells = quad_count * cells_per_quad;
    if (tid >= total_cells || cell_span == 0u)
        return;

    uint32_t quad_index = tid / cells_per_quad;
    uint32_t cell_index = tid - quad_index * cells_per_quad;
    uint32_t cell_y = cell_index / marker_size_with_borders;
    uint32_t cell_x = cell_index - cell_y * marker_size_with_borders;

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

    uint32_t area = cell_span * cell_span;
    uint32_t start_x = cell_x * cell_size + cell_margin_pixels;
    uint32_t start_y = cell_y * cell_size + cell_margin_pixels;
    uint32_t white_pixels = 0u;
    [loop]
    for (uint32_t yy = 0u; yy < cell_span; ++yy)
    {
        [loop]
        for (uint32_t xx = 0u; xx < cell_span; ++xx)
        {
            uint32_t v = uint32_t(sample_warped(
                image_gray,
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

void prepare_decode_quads_buffer(
    uint tid,
    StructuredBuffer<uint8_t> image_gray,
    ImageSize image_size,
    StructuredBuffer<uint32_t> corners,
    RWStructuredBuffer<uint32_t> quad_params_out,
    uint32_t quad_count,
    uint32_t marker_size_with_borders,
    uint32_t cell_size,
    float min_stddev_otsu)
{
    if (tid >= quad_count)
        return;

    uint32_t quad_index = tid;
    uint32_t result_size = marker_size_with_borders * cell_size;

    float h00;
    float h01;
    float h02;
    float h10;
    float h11;
    float h12;
    float h20;
    float h21;
    float h22;
    bool valid_h = build_dst_to_src_homography(
        corners, quad_index, result_size, h00, h01, h02, h10, h11, h12, h20, h21, h22);
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

    uint32_t inner_margin = cell_size / 2u;
    uint32_t inner_start = inner_margin;
    uint32_t inner_end = result_size - inner_margin;
    uint32_t inner_count = 0u;
    float sum = 0.0f;
    float sum_sq = 0.0f;

    [loop]
    for (uint32_t y = inner_start; y < inner_end; ++y)
    {
        [loop]
        for (uint32_t x = inner_start; x < inner_end; ++x)
        {
            float v = float(sample_warped_buffer(image_gray, image_size, h00, h01, h02, h10, h11, h12, h20, h21, h22, x, y));
            sum += v;
            sum_sq += v * v;
            inner_count += 1u;
        }
    }

    float mean = inner_count > 0u ? sum / float(inner_count) : 0.0f;
    float variance = inner_count > 0u ? max(0.0f, sum_sq / float(inner_count) - mean * mean) : 0.0f;
    float stddev = sqrt(variance);

    if (stddev < min_stddev_otsu)
    {
        uint32_t fill = mean > 127.0f ? 1u : 0u;
        write_quad_param_uint(quad_params_out, quad_index, kQuadParamModeWord, 1u);
        write_quad_param_uint(quad_params_out, quad_index, kQuadParamFillBitWord, fill);
        write_quad_param_uint(quad_params_out, quad_index, kQuadParamOtsuThresholdWord, 0u);
        return;
    }

    uint32_t hist[256];
    [unroll]
    for (uint32_t i = 0u; i < 256u; ++i)
    {
        hist[i] = 0u;
    }

    uint32_t total = result_size * result_size;
    float total_sum = 0.0f;
    [loop]
    for (uint32_t y = 0u; y < result_size; ++y)
    {
        [loop]
        for (uint32_t x = 0u; x < result_size; ++x)
        {
            uint32_t v = uint32_t(sample_warped_buffer(image_gray, image_size, h00, h01, h02, h10, h11, h12, h20, h21, h22, x, y));
            hist[v] += 1u;
            total_sum += float(v);
        }
    }

    uint32_t otsu_threshold = 0u;
    float sum_b = 0.0f;
    uint32_t w_b = 0u;
    float max_between = -1.0f;
    [loop]
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

void extract_candidate_bits_buffer(
    uint tid,
    StructuredBuffer<uint8_t> image_gray,
    ImageSize image_size,
    StructuredBuffer<uint32_t> quad_params,
    RWStructuredBuffer<uint32_t> bits_out,
    uint32_t quad_count,
    uint32_t marker_size_with_borders,
    uint32_t cell_size,
    uint32_t cell_margin_pixels,
    uint32_t cell_span)
{
    uint32_t cells_per_quad = marker_size_with_borders * marker_size_with_borders;
    uint32_t total_cells = quad_count * cells_per_quad;
    if (tid >= total_cells || cell_span == 0u)
        return;

    uint32_t quad_index = tid / cells_per_quad;
    uint32_t cell_index = tid - quad_index * cells_per_quad;
    uint32_t cell_y = cell_index / marker_size_with_borders;
    uint32_t cell_x = cell_index - cell_y * marker_size_with_borders;

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

    uint32_t area = cell_span * cell_span;
    uint32_t start_x = cell_x * cell_size + cell_margin_pixels;
    uint32_t start_y = cell_y * cell_size + cell_margin_pixels;
    uint32_t white_pixels = 0u;
    [loop]
    for (uint32_t yy = 0u; yy < cell_span; ++yy)
    {
        [loop]
        for (uint32_t xx = 0u; xx < cell_span; ++xx)
        {
            uint32_t v = uint32_t(sample_warped_buffer(
                image_gray,
                image_size,
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
