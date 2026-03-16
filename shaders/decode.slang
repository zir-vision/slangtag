import "compute/image_buffer.slang";

static const uint32_t kCornersWordsPerQuad = 8u;
static const uint32_t kFittedQuadWordsPerQuad = 15u;
static const uint32_t kFittedQuadCornersStartWord = 3u;
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

[ForceInline]
float read_corner(StructuredBuffer<uint32_t> corners, uint32_t quad_index, uint32_t word_index)
{
    return asfloat(corners[quad_index * kCornersWordsPerQuad + word_index]);
}

[ForceInline]
float read_fitted_quad_corner(StructuredBuffer<uint32_t> fitted_quads, uint32_t quad_index, uint32_t word_index)
{
    return asfloat(fitted_quads[quad_index * kFittedQuadWordsPerQuad + kFittedQuadCornersStartWord + word_index]);
}

bool solve_linear_8x8(inout float a[8][9], out float x[8])
{
    
    for (uint32_t col = 0u; col < 8u; ++col)
    {
        uint32_t pivot = col;
        float best = abs(a[pivot][col]);
        
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
            
            for (uint32_t k = col; k < 9u; ++k)
            {
                float tmp = a[col][k];
                a[col][k] = a[pivot][k];
                a[pivot][k] = tmp;
            }
        }

        float inv = 1.0f / a[col][col];
        
        for (uint32_t k = col; k < 9u; ++k)
        {
            a[col][k] *= inv;
        }

        for (uint32_t row = 0u; row < 8u; ++row)
        {
            if (row == col)
                continue;
            float factor = a[row][col];
            if (abs(factor) <= 1e-12f)
                continue;
            
            for (uint32_t k = col; k < 9u; ++k)
            {
                a[row][k] -= factor * a[col][k];
            }
        }
    }

    [ForceUnroll]
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
    [ForceUnroll]
    for (uint32_t i = 0u; i < 8u; ++i)
    {
        [ForceUnroll]
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

bool build_dst_to_src_homography_from_fitted_quads(StructuredBuffer<uint32_t> fitted_quads,
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

    float u0 = read_fitted_quad_corner(fitted_quads, quad_index, 0u);
    float v0 = read_fitted_quad_corner(fitted_quads, quad_index, 1u);
    float u1 = read_fitted_quad_corner(fitted_quads, quad_index, 2u);
    float v1 = read_fitted_quad_corner(fitted_quads, quad_index, 3u);
    float u2 = read_fitted_quad_corner(fitted_quads, quad_index, 4u);
    float v2 = read_fitted_quad_corner(fitted_quads, quad_index, 5u);
    float u3 = read_fitted_quad_corner(fitted_quads, quad_index, 6u);
    float v3 = read_fitted_quad_corner(fitted_quads, quad_index, 7u);

    float a[8][9];
    [ForceUnroll]
    for (uint32_t i = 0u; i < 8u; ++i)
    {
        [ForceUnroll]
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

