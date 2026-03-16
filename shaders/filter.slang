import "compute/image_buffer.slang";

static const uint32_t kBlobDiffWordsPerPoint = 6u;
static const uint32_t kBlobExtentWordsPerExtent = 11u;
static const uint32_t kSelectedBlobPointWordsPerPoint = 4u;
static const uint32_t kLineFitPointWordsPerPoint = 10u;
static const uint32_t kPeakWordsPerPeak = 3u;
static const uint32_t kPeakExtentWordsPerExtent = 3u;
static const uint32_t kFittedQuadWordsPerQuad = 15u;
static const uint32_t kMaxQuadLocalPeakIndices = 32u;

static const uint32_t kPointRep0Word = 0u;
static const uint32_t kPointRep1Word = 1u;
static const uint32_t kPointXWord = 2u;
static const uint32_t kPointYWord = 3u;
static const uint32_t kPointDxyWord = 4u;
static const uint32_t kPointBlackToWhiteWord = 5u;

static const uint32_t kExtentRep0Word = 0u;
static const uint32_t kExtentRep1Word = 1u;
static const uint32_t kExtentMinXWord = 2u;
static const uint32_t kExtentMaxXWord = 3u;
static const uint32_t kExtentMinYWord = 4u;
static const uint32_t kExtentMaxYWord = 5u;
static const uint32_t kExtentStartingOffsetWord = 6u;
static const uint32_t kExtentCountWord = 7u;
static const uint32_t kExtentPxGxPyGySumWord = 8u;
static const uint32_t kExtentGxSumWord = 9u;
static const uint32_t kExtentGySumWord = 10u;

static const uint32_t kSelectedPointBlobIndexWord = 0u;
static const uint32_t kSelectedPointThetaWord = 1u;
static const uint32_t kSelectedPointXWord = 2u;
static const uint32_t kSelectedPointYWord = 3u;
static const uint32_t kLineFitMxWord = 0u;
static const uint32_t kLineFitMyWord = 1u;
static const uint32_t kLineFitWWord = 2u;
static const uint32_t kLineFitBlobIndexWord = 3u;
static const uint32_t kLineFitMxxLoWord = 4u;
static const uint32_t kLineFitMxxHiWord = 5u;
static const uint32_t kLineFitMyyLoWord = 6u;
static const uint32_t kLineFitMyyHiWord = 7u;
static const uint32_t kLineFitMxyLoWord = 8u;
static const uint32_t kLineFitMxyHiWord = 9u;
static const uint32_t kPeakBlobIndexWord = 0u;
static const uint32_t kPeakErrorWord = 1u;
static const uint32_t kPeakPointIndexWord = 2u;
static const uint32_t kPeakExtentBlobIndexWord = 0u;
static const uint32_t kPeakExtentStartingOffsetWord = 1u;
static const uint32_t kPeakExtentCountWord = 2u;
static const uint32_t kFittedQuadBlobIndexWord = 0u;
static const uint32_t kFittedQuadReversedBorderWord = 1u;
static const uint32_t kFittedQuadScoreWord = 2u;
static const uint32_t kFittedQuadCorner0XWord = 3u;
static const uint32_t kFittedQuadCorner0YWord = 4u;
static const uint32_t kFittedQuadCorner1XWord = 5u;
static const uint32_t kFittedQuadCorner1YWord = 6u;
static const uint32_t kFittedQuadCorner2XWord = 7u;
static const uint32_t kFittedQuadCorner2YWord = 8u;
static const uint32_t kFittedQuadCorner3XWord = 9u;
static const uint32_t kFittedQuadCorner3YWord = 10u;
static const uint32_t kFittedQuadCornerIndex0Word = 11u;
static const uint32_t kFittedQuadCornerIndex1Word = 12u;
static const uint32_t kFittedQuadCornerIndex2Word = 13u;
static const uint32_t kFittedQuadCornerIndex3Word = 14u;

[ForceInline]
uint32_t read_blob_diff_point_word(StructuredBuffer<uint32_t> points, uint32_t point_index, uint32_t word_index)
{
    return points[point_index * kBlobDiffWordsPerPoint + word_index];
}

[ForceInline]
uint32_t read_extent_word<T : IArray<uint32_t>>(T extents, uint32_t extent_index, uint32_t word_index)
{
    return extents[extent_index * kBlobExtentWordsPerExtent + word_index];
}

void dxy_to_gradient(uint32_t dxy, out int gx, out int gy)
{
    // Mirrors the 4 neighborhood offsets used in BlobDiff:
    // 0 -> (+1, 0), 1 -> (+1, +1), 2 -> (0, +1), 3 -> (-1, +1).
    if (dxy == 0u)
    {
        gx = 1;
        gy = 0;
        return;
    }
    if (dxy == 1u)
    {
        gx = 1;
        gy = 1;
        return;
    }
    if (dxy == 2u)
    {
        gx = 0;
        gy = 1;
        return;
    }

    gx = -1;
    gy = 1;
}

[ForceInline]
void write_extent_word(RWStructuredBuffer<uint32_t> extents, uint32_t extent_index, uint32_t word_index, uint32_t value)
{
    extents[extent_index * kBlobExtentWordsPerExtent + word_index] = value;
}

[ForceInline]
void write_selected_blob_point_word(RWStructuredBuffer<uint32_t> points, uint32_t point_index, uint32_t word_index, uint32_t value)
{
    points[point_index * kSelectedBlobPointWordsPerPoint + word_index] = value;
}

[ForceInline]
uint32_t read_selected_blob_point_word(StructuredBuffer<uint32_t> points, uint32_t point_index, uint32_t word_index)
{
    return points[point_index * kSelectedBlobPointWordsPerPoint + word_index];
}

[ForceInline]
void write_line_fit_word(RWStructuredBuffer<uint32_t> points, uint32_t point_index, uint32_t word_index, uint32_t value)
{
    points[point_index * kLineFitPointWordsPerPoint + word_index] = value;
}

[ForceInline]
uint32_t read_line_fit_word(StructuredBuffer<uint32_t> points, uint32_t point_index, uint32_t word_index)
{
    return points[point_index * kLineFitPointWordsPerPoint + word_index];
}

[ForceInline]
void write_peak_word(RWStructuredBuffer<uint32_t> peaks, uint32_t peak_index, uint32_t word_index, uint32_t value)
{
    peaks[peak_index * kPeakWordsPerPeak + word_index] = value;
}

[ForceInline]
uint32_t read_peak_word(StructuredBuffer<uint32_t> peaks, uint32_t peak_index, uint32_t word_index)
{
    return peaks[peak_index * kPeakWordsPerPeak + word_index];
}

[ForceInline]
void write_peak_extent_word(RWStructuredBuffer<uint32_t> peak_extents, uint32_t extent_index, uint32_t word_index, uint32_t value)
{
    peak_extents[extent_index * kPeakExtentWordsPerExtent + word_index] = value;
}

[ForceInline]
void write_fitted_quad_word(RWStructuredBuffer<uint32_t> fitted_quads, uint32_t quad_index, uint32_t word_index, uint32_t value)
{
    fitted_quads[quad_index * kFittedQuadWordsPerQuad + word_index] = value;
}

int64_t read_int64_words(StructuredBuffer<uint32_t> source, uint32_t base_word_index)
{
    uint64_t bits = uint64_t(source[base_word_index]) | (uint64_t(source[base_word_index + 1u]) << 32u);
    return int64_t(bits);
}

void write_int64_words(RWStructuredBuffer<uint32_t> destination, uint32_t base_word_index, int64_t value)
{
    uint64_t bits = uint64_t(value);
    destination[base_word_index] = uint32_t(bits & 0xFFFFFFFFu);
    destination[base_word_index + 1u] = uint32_t(bits >> 32u);
}

float blob_extent_cx(StructuredBuffer<uint32_t> extents, uint32_t extent_index)
{
    uint32_t min_x = read_extent_word(extents, extent_index, kExtentMinXWord);
    uint32_t max_x = read_extent_word(extents, extent_index, kExtentMaxXWord);
    return 0.5f * (float(min_x) + float(max_x)) + 0.05118f;
}

float blob_extent_cy(StructuredBuffer<uint32_t> extents, uint32_t extent_index)
{
    uint32_t min_y = read_extent_word(extents, extent_index, kExtentMinYWord);
    uint32_t max_y = read_extent_word(extents, extent_index, kExtentMaxYWord);
    return 0.5f * (float(min_y) + float(max_y)) - 0.028581f;
}

float blob_extent_dot<T : IArray<uint32_t>>(T extents, uint32_t extent_index)
{
    int pxgx_plus_pygy_sum = asint(read_extent_word(extents, extent_index, kExtentPxGxPyGySumWord));
    int gx_sum = asint(read_extent_word(extents, extent_index, kExtentGxSumWord));
    int gy_sum = asint(read_extent_word(extents, extent_index, kExtentGySumWord));
    int min_x = int(read_extent_word(extents, extent_index, kExtentMinXWord));
    int max_x = int(read_extent_word(extents, extent_index, kExtentMaxXWord));
    int min_y = int(read_extent_word(extents, extent_index, kExtentMinYWord));
    int max_y = int(read_extent_word(extents, extent_index, kExtentMaxYWord));

    return float(pxgx_plus_pygy_sum * 2 - (min_x + max_x) * gx_sum - (min_y + max_y) * gy_sum) * 0.5f
        - 0.05118f * float(gx_sum)
        + 0.028581f * float(gy_sum);
}

float blob_extent_dot_from_moments(
    int pxgx_plus_pygy_sum,
    int gx_sum,
    int gy_sum,
    uint32_t min_x,
    uint32_t max_x,
    uint32_t min_y,
    uint32_t max_y)
{
    return float(pxgx_plus_pygy_sum * 2 - (int(min_x) + int(max_x)) * gx_sum - (int(min_y) + int(max_y)) * gy_sum)
        * 0.5f
        - 0.05118f * float(gx_sum)
        + 0.028581f * float(gy_sum);
}

bool select_blob_extent_from_values(
    uint32_t count,
    uint32_t min_x,
    uint32_t max_x,
    uint32_t min_y,
    uint32_t max_y,
    float extent_dot,
    uint32_t tag_width,
    uint32_t reversed_border,
    uint32_t normal_border,
    uint32_t min_cluster_pixels,
    uint32_t max_cluster_pixels)
{
    uint32_t effective_min_cluster_pixels = max(24u, min_cluster_pixels);
    if (count < effective_min_cluster_pixels)
        return false;
    if (count > max_cluster_pixels)
        return false;

    if ((max_x - min_x) * (max_y - min_y) < tag_width)
        return false;

    bool quad_reversed_border = extent_dot < 0.0f;
    if (reversed_border == 0u && quad_reversed_border)
        return false;
    if (normal_border == 0u && !quad_reversed_border)
        return false;

    return true;
}

bool select_blob_extent<T : IArray<uint32_t>>(T extents,
                        uint32_t extent_index,
                        uint32_t tag_width,
                        uint32_t reversed_border,
                        uint32_t normal_border,
                        uint32_t min_cluster_pixels,
                        uint32_t max_cluster_pixels)
{
    uint32_t count = read_extent_word(extents, extent_index, kExtentCountWord);
    uint32_t effective_min_cluster_pixels = max(24u, min_cluster_pixels);
    if (count < effective_min_cluster_pixels)
        return false;
    if (count > max_cluster_pixels)
        return false;

    uint32_t min_x = read_extent_word(extents, extent_index, kExtentMinXWord);
    uint32_t max_x = read_extent_word(extents, extent_index, kExtentMaxXWord);
    uint32_t min_y = read_extent_word(extents, extent_index, kExtentMinYWord);
    uint32_t max_y = read_extent_word(extents, extent_index, kExtentMaxYWord);
    if ((max_x - min_x) * (max_y - min_y) < tag_width)
        return false;

    bool quad_reversed_border = blob_extent_dot(extents, extent_index) < 0.0f;
    if (reversed_border == 0u && quad_reversed_border)
        return false;
    if (normal_border == 0u && !quad_reversed_border)
        return false;

    return true;
}


void write_line_fit_point(
    RWStructuredBuffer<uint32_t> line_fit_points,
    uint32_t point_index,
    int Mx,
    int My,
    int W,
    uint32_t blob_index,
    int64_t Mxx,
    int64_t Myy,
    int64_t Mxy)
{
    uint32_t point_word_index = point_index * kLineFitPointWordsPerPoint;
    write_line_fit_word(line_fit_points, point_index, kLineFitMxWord, asuint(Mx));
    write_line_fit_word(line_fit_points, point_index, kLineFitMyWord, asuint(My));
    write_line_fit_word(line_fit_points, point_index, kLineFitWWord, asuint(W));
    write_line_fit_word(line_fit_points, point_index, kLineFitBlobIndexWord, blob_index);
    write_int64_words(line_fit_points, point_word_index + kLineFitMxxLoWord, Mxx);
    write_int64_words(line_fit_points, point_word_index + kLineFitMyyLoWord, Myy);
    write_int64_words(line_fit_points, point_word_index + kLineFitMxyLoWord, Mxy);
}

void read_line_fit_point(
    StructuredBuffer<uint32_t> line_fit_points,
    uint32_t point_index,
    out int Mx,
    out int My,
    out int W,
    out uint32_t blob_index,
    out int64_t Mxx,
    out int64_t Myy,
    out int64_t Mxy)
{
    uint32_t point_word_index = point_index * kLineFitPointWordsPerPoint;
    Mx = asint(read_line_fit_word(line_fit_points, point_index, kLineFitMxWord));
    My = asint(read_line_fit_word(line_fit_points, point_index, kLineFitMyWord));
    W = asint(read_line_fit_word(line_fit_points, point_index, kLineFitWWord));
    blob_index = read_line_fit_word(line_fit_points, point_index, kLineFitBlobIndexWord);
    Mxx = read_int64_words(line_fit_points, point_word_index + kLineFitMxxLoWord);
    Myy = read_int64_words(line_fit_points, point_word_index + kLineFitMyyLoWord);
    Mxy = read_int64_words(line_fit_points, point_word_index + kLineFitMxyLoWord);
}

float fit_line_error(int N, int Mx, int My, int64_t Mxx, int64_t Myy, int64_t Mxy, int W)
{
    if (W <= 0)
        return 0.0f;

    float Cxx = float(Mxx * int64_t(W) - int64_t(Mx) * int64_t(Mx));
    float Cxy = float(Mxy * int64_t(W) - int64_t(Mx) * int64_t(My));
    float Cyy = float(Myy * int64_t(W) - int64_t(My) * int64_t(My));

    float eig_small = ((Cxx + Cyy) - sqrt((Cxx - Cyy) * (Cxx - Cyy) + 4.0f * Cxy * Cxy))
        / float(int64_t(W) * int64_t(W) * int64_t(8));
    return float(N) * eig_small;
}

float line_fit_filter_coefficient(int i)
{
    if (i == 0)
        return 0.0111089963f;
    if (i == 1)
        return 0.1353352815f;
    if (i == 2)
        return 0.6065306663f;
    if (i == 3)
        return 1.0f;
    if (i == 4)
        return 0.6065306663f;
    if (i == 5)
        return 0.1353352815f;
    return 0.0111089963f;
}
