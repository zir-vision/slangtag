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

uint32_t read_blob_diff_point_word(StructuredBuffer<uint32_t> points, uint32_t point_index, uint32_t word_index)
{
    return points[point_index * kBlobDiffWordsPerPoint + word_index];
}

uint32_t read_extent_word(StructuredBuffer<uint32_t> extents, uint32_t extent_index, uint32_t word_index)
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

void write_extent_word(RWStructuredBuffer<uint32_t> extents, uint32_t extent_index, uint32_t word_index, uint32_t value)
{
    extents[extent_index * kBlobExtentWordsPerExtent + word_index] = value;
}

void write_selected_blob_point_word(RWStructuredBuffer<uint32_t> points, uint32_t point_index, uint32_t word_index, uint32_t value)
{
    points[point_index * kSelectedBlobPointWordsPerPoint + word_index] = value;
}

uint32_t read_selected_blob_point_word(StructuredBuffer<uint32_t> points, uint32_t point_index, uint32_t word_index)
{
    return points[point_index * kSelectedBlobPointWordsPerPoint + word_index];
}

void write_line_fit_word(RWStructuredBuffer<uint32_t> points, uint32_t point_index, uint32_t word_index, uint32_t value)
{
    points[point_index * kLineFitPointWordsPerPoint + word_index] = value;
}

uint32_t read_line_fit_word(StructuredBuffer<uint32_t> points, uint32_t point_index, uint32_t word_index)
{
    return points[point_index * kLineFitPointWordsPerPoint + word_index];
}

void write_peak_word(RWStructuredBuffer<uint32_t> peaks, uint32_t peak_index, uint32_t word_index, uint32_t value)
{
    peaks[peak_index * kPeakWordsPerPeak + word_index] = value;
}

uint32_t read_peak_word(StructuredBuffer<uint32_t> peaks, uint32_t peak_index, uint32_t word_index)
{
    return peaks[peak_index * kPeakWordsPerPeak + word_index];
}

void write_peak_extent_word(RWStructuredBuffer<uint32_t> peak_extents, uint32_t extent_index, uint32_t word_index, uint32_t value)
{
    peak_extents[extent_index * kPeakExtentWordsPerExtent + word_index] = value;
}

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

void build_blob_pair_extents(uint tid,
                             StructuredBuffer<uint32_t> sorted_points,
                             RWStructuredBuffer<uint32_t> extents_out,
                             RWStructuredBuffer<uint32_t> extent_count_out,
                             uint32_t valid_points)
{
    if (tid != 0u)
        return;

    uint32_t extent_count = 0u;
    uint32_t i = 0u;

    while (i < valid_points)
    {
        uint32_t rep0 = read_blob_diff_point_word(sorted_points, i, kPointRep0Word);
        uint32_t rep1 = read_blob_diff_point_word(sorted_points, i, kPointRep1Word);

        uint32_t min_x = read_blob_diff_point_word(sorted_points, i, kPointXWord);
        uint32_t max_x = min_x;
        uint32_t min_y = read_blob_diff_point_word(sorted_points, i, kPointYWord);
        uint32_t max_y = min_y;
        uint32_t start = i;
        uint32_t count = 0u;

        int pxgx_plus_pygy_sum = 0;
        int gx_sum = 0;
        int gy_sum = 0;

        while (i < valid_points)
        {
            uint32_t point_rep0 = read_blob_diff_point_word(sorted_points, i, kPointRep0Word);
            uint32_t point_rep1 = read_blob_diff_point_word(sorted_points, i, kPointRep1Word);
            if (point_rep0 != rep0 || point_rep1 != rep1)
                break;

            uint32_t x = read_blob_diff_point_word(sorted_points, i, kPointXWord);
            uint32_t y = read_blob_diff_point_word(sorted_points, i, kPointYWord);
            if (x < min_x)
                min_x = x;
            if (x > max_x)
                max_x = x;
            if (y < min_y)
                min_y = y;
            if (y > max_y)
                max_y = y;

            int gx;
            int gy;
            dxy_to_gradient(read_blob_diff_point_word(sorted_points, i, kPointDxyWord), gx, gy);

            if (read_blob_diff_point_word(sorted_points, i, kPointBlackToWhiteWord) == 0u)
            {
                gx = -gx;
                gy = -gy;
            }

            pxgx_plus_pygy_sum += int(x) * gx + int(y) * gy;
            gx_sum += gx;
            gy_sum += gy;
            count += 1u;
            i += 1u;
        }

        write_extent_word(extents_out, extent_count, kExtentRep0Word, rep0);
        write_extent_word(extents_out, extent_count, kExtentRep1Word, rep1);
        write_extent_word(extents_out, extent_count, kExtentMinXWord, min_x);
        write_extent_word(extents_out, extent_count, kExtentMaxXWord, max_x);
        write_extent_word(extents_out, extent_count, kExtentMinYWord, min_y);
        write_extent_word(extents_out, extent_count, kExtentMaxYWord, max_y);
        write_extent_word(extents_out, extent_count, kExtentStartingOffsetWord, start);
        write_extent_word(extents_out, extent_count, kExtentCountWord, count);
        write_extent_word(extents_out, extent_count, kExtentPxGxPyGySumWord, asuint(pxgx_plus_pygy_sum));
        write_extent_word(extents_out, extent_count, kExtentGxSumWord, asuint(gx_sum));
        write_extent_word(extents_out, extent_count, kExtentGySumWord, asuint(gy_sum));

        extent_count += 1u;
    }

    extent_count_out[0] = extent_count;
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

float blob_extent_dot(StructuredBuffer<uint32_t> extents, uint32_t extent_index)
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

bool select_blob_extent(StructuredBuffer<uint32_t> extents,
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

void filter_blob_pair_extents(uint tid,
                              StructuredBuffer<uint32_t> extents_in,
                              RWStructuredBuffer<uint32_t> extents_out,
                              RWStructuredBuffer<uint32_t> selected_extent_count_out,
                              RWStructuredBuffer<uint32_t> selected_point_count_out,
                              uint32_t extent_count,
                              uint32_t tag_width,
                              uint32_t reversed_border,
                              uint32_t normal_border,
                              uint32_t min_cluster_pixels,
                              uint32_t max_cluster_pixels)
{
    if (tid != 0u)
        return;

    uint32_t selected_extent_count = 0u;
    uint32_t selected_point_count = 0u;

    for (uint32_t extent_index = 0u; extent_index < extent_count; ++extent_index)
    {
        [ForceUnroll]
        for (uint32_t word_index = 0u; word_index < kBlobExtentWordsPerExtent; ++word_index)
        {
            write_extent_word(
                extents_out,
                extent_index,
                word_index,
                read_extent_word(extents_in, extent_index, word_index));
        }

        bool selected = select_blob_extent(
            extents_in,
            extent_index,
            tag_width,
            reversed_border,
            normal_border,
            min_cluster_pixels,
            max_cluster_pixels);

        uint32_t count = selected ? read_extent_word(extents_in, extent_index, kExtentCountWord) : 0u;

        write_extent_word(extents_out, extent_index, kExtentStartingOffsetWord, selected_point_count);
        write_extent_word(extents_out, extent_index, kExtentCountWord, count);

        selected_point_count += count;
        if (selected)
            selected_extent_count += 1u;
    }

    selected_extent_count_out[0] = selected_extent_count;
    selected_point_count_out[0] = selected_point_count;
}

uint32_t find_extent_index_for_point(StructuredBuffer<uint32_t> extents,
                                     uint32_t extent_count,
                                     uint32_t point_index)
{
    uint32_t left = 0u;
    uint32_t right = extent_count;

    while (left < right)
    {
        uint32_t mid = left + (right - left) / 2u;
        uint32_t start = read_extent_word(extents, mid, kExtentStartingOffsetWord);
        uint32_t count = read_extent_word(extents, mid, kExtentCountWord);
        uint32_t end = start + count;

        if (point_index < start)
        {
            right = mid;
            continue;
        }
        if (point_index >= end)
        {
            left = mid + 1u;
            continue;
        }
        return mid;
    }

    return 0xFFFFFFFFu;
}

uint32_t point_theta_from_extent(StructuredBuffer<uint32_t> extents,
                                 uint32_t extent_index,
                                 uint32_t x,
                                 uint32_t y)
{
    static const float kPi = 3.14159265358979323846f;
    static const float kThetaScale = 8.0e6f;

    float cx = blob_extent_cx(extents, extent_index);
    float cy = blob_extent_cy(extents, extent_index);

    float theta = (atan2(float(y) - cy, float(x) - cx) + kPi) * kThetaScale;
    return uint32_t(max(0.0f, theta) + 0.5f);
}

void rewrite_selected_blob_points_with_theta(
    uint tid,
    StructuredBuffer<uint32_t> sorted_points,
    StructuredBuffer<uint32_t> extents_in,
    StructuredBuffer<uint32_t> filtered_extents,
    RWStructuredBuffer<uint32_t> selected_points_out,
    uint32_t extent_count,
    uint32_t valid_points)
{
    if (tid >= valid_points)
        return;

    uint32_t extent_index = find_extent_index_for_point(extents_in, extent_count, tid);
    if (extent_index == 0xFFFFFFFFu)
        return;

    uint32_t selected_count = read_extent_word(filtered_extents, extent_index, kExtentCountWord);
    if (selected_count == 0u)
        return;

    uint32_t original_start = read_extent_word(extents_in, extent_index, kExtentStartingOffsetWord);
    uint32_t local_offset = tid - original_start;
    if (local_offset >= selected_count)
        return;

    uint32_t selected_start = read_extent_word(filtered_extents, extent_index, kExtentStartingOffsetWord);
    uint32_t selected_index = selected_start + local_offset;

    uint32_t x = read_blob_diff_point_word(sorted_points, tid, kPointXWord);
    uint32_t y = read_blob_diff_point_word(sorted_points, tid, kPointYWord);

    write_selected_blob_point_word(
        selected_points_out,
        selected_index,
        kSelectedPointBlobIndexWord,
        extent_index);
    write_selected_blob_point_word(
        selected_points_out,
        selected_index,
        kSelectedPointThetaWord,
        point_theta_from_extent(extents_in, extent_index, x, y));
    write_selected_blob_point_word(
        selected_points_out,
        selected_index,
        kSelectedPointXWord,
        x);
    write_selected_blob_point_word(
        selected_points_out,
        selected_index,
        kSelectedPointYWord,
        y);
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

void build_line_fit_points(
    uint tid,
    StructuredBuffer<uint32_t> sorted_selected_points,
    Texture2D<uint8_t1> decimated_image,
    RWStructuredBuffer<uint32_t> line_fit_points_out,
    uint32_t point_count,
    uint32_t decimated_width,
    uint32_t decimated_height,
    uint32_t decimate)
{
    if (tid != 0u)
        return;

    uint32_t previous_blob_index = 0xFFFFFFFFu;
    int running_Mx = 0;
    int running_My = 0;
    int running_W = 0;
    int64_t running_Mxx = 0;
    int64_t running_Myy = 0;
    int64_t running_Mxy = 0;

    for (uint32_t i = 0u; i < point_count; ++i)
    {
        uint32_t blob_index = read_selected_blob_point_word(sorted_selected_points, i, kSelectedPointBlobIndexWord);
        uint32_t x = read_selected_blob_point_word(sorted_selected_points, i, kSelectedPointXWord);
        uint32_t y = read_selected_blob_point_word(sorted_selected_points, i, kSelectedPointYWord);

        int delta = int(decimate) - 1;
        // In the CUDA path, IndexPoint::x/y are already fixed-point values.
        // This port stores integer pixel coordinates, so reconstruct fixed-point
        // coordinates before line fitting.
        int ix2 = int(x) * int(decimate) + delta;
        int iy2 = int(y) * int(decimate) + delta;
        int ix = ix2 / int(decimate);
        int iy = iy2 / int(decimate);

        int W = 1;
        if (ix > 0 && uint32_t(ix + 1) < decimated_width && iy > 0 && uint32_t(iy + 1) < decimated_height)
        {
            int grad_x = int(decimated_image[uint2(uint32_t(ix + 1), uint32_t(iy))].x)
                - int(decimated_image[uint2(uint32_t(ix - 1), uint32_t(iy))].x);
            int grad_y = int(decimated_image[uint2(uint32_t(ix), uint32_t(iy + 1))].x)
                - int(decimated_image[uint2(uint32_t(ix), uint32_t(iy - 1))].x);

            W = int(sqrt(float(grad_x * grad_x + grad_y * grad_y))) + 1;
        }

        int point_Mx = W * ix2;
        int point_My = W * iy2;
        int64_t point_Mxx = int64_t(W) * int64_t(ix2) * int64_t(ix2);
        int64_t point_Mxy = int64_t(W) * int64_t(ix2) * int64_t(iy2);
        int64_t point_Myy = int64_t(W) * int64_t(iy2) * int64_t(iy2);

        if (blob_index != previous_blob_index)
        {
            running_Mx = 0;
            running_My = 0;
            running_W = 0;
            running_Mxx = 0;
            running_Myy = 0;
            running_Mxy = 0;
            previous_blob_index = blob_index;
        }

        running_Mx += point_Mx;
        running_My += point_My;
        running_W += W;
        running_Mxx += point_Mxx;
        running_Myy += point_Myy;
        running_Mxy += point_Mxy;

        write_line_fit_point(
            line_fit_points_out,
            i,
            running_Mx,
            running_My,
            running_W,
            blob_index,
            running_Mxx,
            running_Myy,
            running_Mxy);
    }
}

void build_line_fit_points_buffer(
    uint tid,
    StructuredBuffer<uint32_t> sorted_selected_points,
    StructuredBuffer<uint8_t> decimated_image,
    ImageSize decimated_size,
    RWStructuredBuffer<uint32_t> line_fit_points_out,
    uint32_t point_count,
    uint32_t decimate)
{
    if (tid != 0u)
        return;

    uint32_t previous_blob_index = 0xFFFFFFFFu;
    int running_Mx = 0;
    int running_My = 0;
    int running_W = 0;
    int64_t running_Mxx = 0;
    int64_t running_Myy = 0;
    int64_t running_Mxy = 0;

    for (uint32_t i = 0u; i < point_count; ++i)
    {
        uint32_t blob_index = read_selected_blob_point_word(sorted_selected_points, i, kSelectedPointBlobIndexWord);
        uint32_t x = read_selected_blob_point_word(sorted_selected_points, i, kSelectedPointXWord);
        uint32_t y = read_selected_blob_point_word(sorted_selected_points, i, kSelectedPointYWord);

        int delta = int(decimate) - 1;
        int ix2 = int(x) * int(decimate) + delta;
        int iy2 = int(y) * int(decimate) + delta;
        int ix = ix2 / int(decimate);
        int iy = iy2 / int(decimate);

        int W = 1;
        if (ix > 0 && uint32_t(ix + 1) < decimated_size.width && iy > 0 && uint32_t(iy + 1) < decimated_size.height)
        {
            int grad_x = int(image_u8_read(decimated_image, decimated_size, uint2(uint32_t(ix + 1), uint32_t(iy))))
                - int(image_u8_read(decimated_image, decimated_size, uint2(uint32_t(ix - 1), uint32_t(iy))));
            int grad_y = int(image_u8_read(decimated_image, decimated_size, uint2(uint32_t(ix), uint32_t(iy + 1))))
                - int(image_u8_read(decimated_image, decimated_size, uint2(uint32_t(ix), uint32_t(iy - 1))));

            W = int(sqrt(float(grad_x * grad_x + grad_y * grad_y))) + 1;
        }

        int point_Mx = W * ix2;
        int point_My = W * iy2;
        int64_t point_Mxx = int64_t(W) * int64_t(ix2) * int64_t(ix2);
        int64_t point_Mxy = int64_t(W) * int64_t(ix2) * int64_t(iy2);
        int64_t point_Myy = int64_t(W) * int64_t(iy2) * int64_t(iy2);

        if (blob_index != previous_blob_index)
        {
            running_Mx = 0;
            running_My = 0;
            running_W = 0;
            running_Mxx = 0;
            running_Myy = 0;
            running_Mxy = 0;
            previous_blob_index = blob_index;
        }

        running_Mx += point_Mx;
        running_My += point_My;
        running_W += W;
        running_Mxx += point_Mxx;
        running_Myy += point_Myy;
        running_Mxy += point_Mxy;

        write_line_fit_point(
            line_fit_points_out,
            i,
            running_Mx,
            running_My,
            running_W,
            blob_index,
            running_Mxx,
            running_Myy,
            running_Mxy);
    }
}

void fit_line_errors_and_peaks(
    uint tid,
    StructuredBuffer<uint32_t> line_fit_points,
    StructuredBuffer<uint32_t> filtered_extents,
    RWStructuredBuffer<uint32_t> errs_out,
    RWStructuredBuffer<uint32_t> filtered_errs_out,
    RWStructuredBuffer<uint32_t> peaks_out,
    uint32_t extent_count,
    uint32_t point_count)
{
    if (tid != 0u)
        return;

    for (uint32_t i = 0u; i < point_count; ++i)
    {
        errs_out[i] = asuint(0.0f);
        filtered_errs_out[i] = asuint(0.0f);
        write_peak_word(peaks_out, i, kPeakBlobIndexWord, 0xFFFFu);
        write_peak_word(peaks_out, i, kPeakErrorWord, asuint(0.0f));
        write_peak_word(peaks_out, i, kPeakPointIndexWord, i);
    }

    for (uint32_t i = 0u; i < point_count; ++i)
    {
        int Mx_dummy;
        int My_dummy;
        int W_dummy;
        uint32_t blob_index;
        int64_t Mxx_dummy;
        int64_t Myy_dummy;
        int64_t Mxy_dummy;
        read_line_fit_point(line_fit_points, i, Mx_dummy, My_dummy, W_dummy, blob_index, Mxx_dummy, Myy_dummy, Mxy_dummy);

        if (blob_index >= extent_count)
            continue;

        uint32_t start = read_extent_word(filtered_extents, blob_index, kExtentStartingOffsetWord);
        uint32_t count = read_extent_word(filtered_extents, blob_index, kExtentCountWord);
        if (count == 0u || i < start || i >= start + count)
            continue;

        uint32_t local_index = i - start;
        uint32_t ksz = min(20u, count / 12u);
        uint32_t i0 = (local_index + 2u * count - ksz) % count;
        uint32_t i1 = (local_index + count + ksz) % count;

        int Mx;
        int My;
        int W;
        int64_t Mxx;
        int64_t Myy;
        int64_t Mxy;
        int N;

        if (i0 < i1)
        {
            N = int(i1 - i0 + 1u);

            int Mx1;
            int My1;
            int W1;
            uint32_t blob1;
            int64_t Mxx1;
            int64_t Myy1;
            int64_t Mxy1;
            read_line_fit_point(line_fit_points, start + i1, Mx1, My1, W1, blob1, Mxx1, Myy1, Mxy1);

            Mx = Mx1;
            My = My1;
            W = W1;
            Mxx = Mxx1;
            Myy = Myy1;
            Mxy = Mxy1;

            if (i0 > 0u)
            {
                int Mx0;
                int My0;
                int W0;
                uint32_t blob0;
                int64_t Mxx0;
                int64_t Myy0;
                int64_t Mxy0;
                read_line_fit_point(line_fit_points, start + i0 - 1u, Mx0, My0, W0, blob0, Mxx0, Myy0, Mxy0);
                Mx -= Mx0;
                My -= My0;
                W -= W0;
                Mxx -= Mxx0;
                Myy -= Myy0;
                Mxy -= Mxy0;
            }
        }
        else
        {
            int Mx0;
            int My0;
            int W0;
            uint32_t blob0;
            int64_t Mxx0;
            int64_t Myy0;
            int64_t Mxy0;
            read_line_fit_point(line_fit_points, start + i0 - 1u, Mx0, My0, W0, blob0, Mxx0, Myy0, Mxy0);

            int MxN;
            int MyN;
            int WN;
            uint32_t blobN;
            int64_t MxxN;
            int64_t MyyN;
            int64_t MxyN;
            read_line_fit_point(line_fit_points, start + count - 1u, MxN, MyN, WN, blobN, MxxN, MyyN, MxyN);

            int Mx1;
            int My1;
            int W1;
            uint32_t blob1;
            int64_t Mxx1;
            int64_t Myy1;
            int64_t Mxy1;
            read_line_fit_point(line_fit_points, start + i1, Mx1, My1, W1, blob1, Mxx1, Myy1, Mxy1);

            Mx = (MxN - Mx0) + Mx1;
            My = (MyN - My0) + My1;
            W = (WN - W0) + W1;
            Mxx = (MxxN - Mxx0) + Mxx1;
            Myy = (MyyN - Myy0) + Myy1;
            Mxy = (MxyN - Mxy0) + Mxy1;
            N = int(count - i0 + i1 + 1u);
        }

        float err = fit_line_error(N, Mx, My, Mxx, Myy, Mxy, W);
        errs_out[i] = asuint(err);
    }

    for (uint32_t i = 0u; i < point_count; ++i)
    {
        int Mx_dummy;
        int My_dummy;
        int W_dummy;
        uint32_t blob_index;
        int64_t Mxx_dummy;
        int64_t Myy_dummy;
        int64_t Mxy_dummy;
        read_line_fit_point(line_fit_points, i, Mx_dummy, My_dummy, W_dummy, blob_index, Mxx_dummy, Myy_dummy, Mxy_dummy);

        if (blob_index >= extent_count)
            continue;

        uint32_t start = read_extent_word(filtered_extents, blob_index, kExtentStartingOffsetWord);
        uint32_t count = read_extent_word(filtered_extents, blob_index, kExtentCountWord);
        if (count == 0u || i < start || i >= start + count)
            continue;

        int local_index = int(i - start);
        float filtered_err = 0.0f;
        for (int j = 0; j < 7; ++j)
        {
            int neighbor_local_index = (local_index + int(count) + j - 3) % int(count);
            uint32_t neighbor_index = start + uint32_t(neighbor_local_index);
            filtered_err += asfloat(errs_out[neighbor_index]) * line_fit_filter_coefficient(j);
        }

        filtered_errs_out[i] = asuint(filtered_err);
    }

    for (uint32_t i = 0u; i < point_count; ++i)
    {
        int Mx_dummy;
        int My_dummy;
        int W_dummy;
        uint32_t blob_index;
        int64_t Mxx_dummy;
        int64_t Myy_dummy;
        int64_t Mxy_dummy;
        read_line_fit_point(line_fit_points, i, Mx_dummy, My_dummy, W_dummy, blob_index, Mxx_dummy, Myy_dummy, Mxy_dummy);

        if (blob_index >= extent_count)
            continue;

        uint32_t start = read_extent_word(filtered_extents, blob_index, kExtentStartingOffsetWord);
        uint32_t count = read_extent_word(filtered_extents, blob_index, kExtentCountWord);
        if (count == 0u || i < start || i >= start + count)
            continue;

        uint32_t local_index = i - start;
        uint32_t before_index = start + ((local_index + count - 1u) % count);
        uint32_t after_index = start + ((local_index + 1u) % count);

        float before_error = asfloat(filtered_errs_out[before_index]);
        float my_error = asfloat(filtered_errs_out[i]);
        float after_error = asfloat(filtered_errs_out[after_index]);
        bool is_peak = my_error > before_error && my_error > after_error;

        write_peak_word(peaks_out, i, kPeakBlobIndexWord, is_peak ? blob_index : 0xFFFFu);
        write_peak_word(peaks_out, i, kPeakErrorWord, asuint(-my_error));
        write_peak_word(peaks_out, i, kPeakPointIndexWord, i);
    }
}

void build_peak_extents(
    uint tid,
    StructuredBuffer<uint32_t> sorted_peaks,
    RWStructuredBuffer<uint32_t> peak_extents_out,
    RWStructuredBuffer<uint32_t> peak_extent_count_out,
    uint32_t valid_peaks)
{
    if (tid != 0u)
        return;

    if (valid_peaks == 0u)
    {
        peak_extent_count_out[0] = 0u;
        return;
    }

    uint32_t peak_extent_count = 0u;
    uint32_t i = 0u;
    while (i < valid_peaks)
    {
        uint32_t blob_index = read_peak_word(sorted_peaks, i, kPeakBlobIndexWord);
        uint32_t start = i;
        uint32_t count = 1u;

        while (i + count < valid_peaks &&
               read_peak_word(sorted_peaks, i + count, kPeakBlobIndexWord) == blob_index)
        {
            count += 1u;
        }

        write_peak_extent_word(peak_extents_out, peak_extent_count, kPeakExtentBlobIndexWord, blob_index);
        write_peak_extent_word(peak_extents_out, peak_extent_count, kPeakExtentStartingOffsetWord, start);
        write_peak_extent_word(peak_extents_out, peak_extent_count, kPeakExtentCountWord, count);

        peak_extent_count += 1u;
        i += count;
    }

    peak_extent_count_out[0] = peak_extent_count;
}

struct QuadLineFitMoments
{
    int N;
    int Mx;
    int My;
    int W;
    int64_t Mxx;
    int64_t Myy;
    int64_t Mxy;
};

void read_line_fit_prefix_sums(
    StructuredBuffer<uint32_t> line_fit_points,
    uint32_t index,
    out int Mx,
    out int My,
    out int W,
    out int64_t Mxx,
    out int64_t Myy,
    out int64_t Mxy)
{
    int dummy_Mx;
    int dummy_My;
    int dummy_W;
    uint32_t dummy_blob_index;
    read_line_fit_point(
        line_fit_points,
        index,
        dummy_Mx,
        dummy_My,
        dummy_W,
        dummy_blob_index,
        Mxx,
        Myy,
        Mxy);

    Mx = dummy_Mx;
    My = dummy_My;
    W = dummy_W;
}

void range_non_wrap_moments(
    StructuredBuffer<uint32_t> line_fit_points,
    uint32_t blob_start,
    uint32_t local_start,
    uint32_t local_end,
    out QuadLineFitMoments moments_out)
{
    int end_Mx;
    int end_My;
    int end_W;
    int64_t end_Mxx;
    int64_t end_Myy;
    int64_t end_Mxy;
    read_line_fit_prefix_sums(
        line_fit_points,
        blob_start + local_end,
        end_Mx,
        end_My,
        end_W,
        end_Mxx,
        end_Myy,
        end_Mxy);

    if (local_start == 0u)
    {
        moments_out.N = int(local_end + 1u);
        moments_out.Mx = end_Mx;
        moments_out.My = end_My;
        moments_out.W = end_W;
        moments_out.Mxx = end_Mxx;
        moments_out.Myy = end_Myy;
        moments_out.Mxy = end_Mxy;
        return;
    }

    int start_Mx;
    int start_My;
    int start_W;
    int64_t start_Mxx;
    int64_t start_Myy;
    int64_t start_Mxy;
    read_line_fit_prefix_sums(
        line_fit_points,
        blob_start + local_start - 1u,
        start_Mx,
        start_My,
        start_W,
        start_Mxx,
        start_Myy,
        start_Mxy);

    moments_out.N = int(local_end - local_start + 1u);
    moments_out.Mx = end_Mx - start_Mx;
    moments_out.My = end_My - start_My;
    moments_out.W = end_W - start_W;
    moments_out.Mxx = end_Mxx - start_Mxx;
    moments_out.Myy = end_Myy - start_Myy;
    moments_out.Mxy = end_Mxy - start_Mxy;
}

void segment_moments(
    StructuredBuffer<uint32_t> line_fit_points,
    uint32_t blob_start,
    uint32_t blob_count,
    uint32_t local_start,
    uint32_t local_end,
    out QuadLineFitMoments moments_out)
{
    if (local_start <= local_end)
    {
        range_non_wrap_moments(
            line_fit_points,
            blob_start,
            local_start,
            local_end,
            moments_out);
        return;
    }

    QuadLineFitMoments sums_a;
    QuadLineFitMoments sums_b;
    range_non_wrap_moments(
        line_fit_points,
        blob_start,
        local_start,
        blob_count - 1u,
        sums_a);
    range_non_wrap_moments(
        line_fit_points,
        blob_start,
        0u,
        local_end,
        sums_b);

    moments_out.N = int((blob_count - local_start) + (local_end + 1u));
    moments_out.Mx = sums_a.Mx + sums_b.Mx;
    moments_out.My = sums_a.My + sums_b.My;
    moments_out.W = sums_a.W + sums_b.W;
    moments_out.Mxx = sums_a.Mxx + sums_b.Mxx;
    moments_out.Myy = sums_a.Myy + sums_b.Myy;
    moments_out.Mxy = sums_a.Mxy + sums_b.Mxy;
}

float line_fit_mse(QuadLineFitMoments moments)
{
    if (moments.W <= 0)
        return 0.0f;

    float Cxx = float(moments.Mxx * int64_t(moments.W) - int64_t(moments.Mx) * int64_t(moments.Mx));
    float Cxy = float(moments.Mxy * int64_t(moments.W) - int64_t(moments.Mx) * int64_t(moments.My));
    float Cyy = float(moments.Myy * int64_t(moments.W) - int64_t(moments.My) * int64_t(moments.My));
    float hypot_cached = sqrt((Cxx - Cyy) * (Cxx - Cyy) + 4.0f * Cxy * Cxy);
    return ((Cxx + Cyy) - hypot_cached)
        / float(int64_t(moments.W) * int64_t(moments.W) * int64_t(8));
}

bool fit_line_from_moments(QuadLineFitMoments moments, out float4 line_out)
{
    if (moments.W <= 0)
        return false;

    float Cxx = float(moments.Mxx * int64_t(moments.W) - int64_t(moments.Mx) * int64_t(moments.Mx));
    float Cxy = float(moments.Mxy * int64_t(moments.W) - int64_t(moments.Mx) * int64_t(moments.My));
    float Cyy = float(moments.Myy * int64_t(moments.W) - int64_t(moments.My) * int64_t(moments.My));
    float hypot_cached = sqrt((Cxx - Cyy) * (Cxx - Cyy) + 4.0f * Cxy * Cxy);

    float nx1 = (Cxx - Cyy) - hypot_cached;
    float ny1 = 2.0f * Cxy;
    float nx2 = 2.0f * Cxy;
    float ny2 = (Cyy - Cxx) - hypot_cached;

    float nx;
    float ny;
    if (nx1 * nx1 + ny1 * ny1 > nx2 * nx2 + ny2 * ny2)
    {
        nx = nx1;
        ny = ny1;
    }
    else
    {
        nx = nx2;
        ny = ny2;
    }

    float len = sqrt(nx * nx + ny * ny);
    if (len < 1.0e-12f)
        return false;

    line_out.x = float(moments.Mx) / float(moments.W * 2);
    line_out.y = float(moments.My) / float(moments.W * 2);
    line_out.z = nx / len;
    line_out.w = ny / len;
    return true;
}

float triangle_area(float2 a, float2 b, float2 c)
{
    float ab = length(b - a);
    float bc = length(c - b);
    float ca = length(a - c);
    float p = 0.5f * (ab + bc + ca);
    return sqrt(max(0.0f, p * (p - ab) * (p - bc) * (p - ca)));
}

void fit_quads(
    uint tid,
    StructuredBuffer<uint32_t> sorted_peaks,
    StructuredBuffer<uint32_t> peak_extents,
    StructuredBuffer<uint32_t> line_fit_points,
    StructuredBuffer<uint32_t> filtered_blob_extents,
    RWStructuredBuffer<uint32_t> fitted_quads_out,
    RWStructuredBuffer<uint32_t> fitted_quad_count_out,
    uint32_t peak_extent_count,
    uint32_t filtered_blob_extent_count,
    uint32_t max_nmaxima,
    float max_line_fit_mse,
    float cos_critical_rad,
    uint32_t min_tag_width,
    float quad_decimate)
{
    if (tid >= peak_extent_count)
        return;

    uint32_t peak_extent_index = tid;
    uint32_t blob_index = peak_extents[peak_extent_index * kPeakExtentWordsPerExtent + kPeakExtentBlobIndexWord];
    uint32_t peak_start = peak_extents[peak_extent_index * kPeakExtentWordsPerExtent + kPeakExtentStartingOffsetWord];
    uint32_t peak_count = peak_extents[peak_extent_index * kPeakExtentWordsPerExtent + kPeakExtentCountWord];

    if (blob_index >= filtered_blob_extent_count || peak_count < 4u)
        return;

    uint32_t blob_start = read_extent_word(filtered_blob_extents, blob_index, kExtentStartingOffsetWord);
    uint32_t blob_count = read_extent_word(filtered_blob_extents, blob_index, kExtentCountWord);
    if (blob_count < 8u)
        return;

    uint32_t local_peak_indices[kMaxQuadLocalPeakIndices];
    uint32_t local_peak_count = 0u;
    uint32_t maxima_to_use = min(max_nmaxima, peak_count);
    maxima_to_use = min(maxima_to_use, kMaxQuadLocalPeakIndices);

    for (uint32_t i = 0u; i < maxima_to_use; ++i)
    {
        uint32_t point_index = read_peak_word(sorted_peaks, peak_start + i, kPeakPointIndexWord);
        if (point_index < blob_start)
            continue;

        uint32_t local_index = point_index - blob_start;
        if (local_index >= blob_count)
            continue;

        bool seen = false;
        for (uint32_t j = 0u; j < local_peak_count; ++j)
        {
            if (local_peak_indices[j] == local_index)
            {
                seen = true;
                break;
            }
        }
        if (seen)
            continue;

        local_peak_indices[local_peak_count] = local_index;
        local_peak_count += 1u;
    }

    for (uint32_t i = 1u; i < local_peak_count; ++i)
    {
        uint32_t key = local_peak_indices[i];
        uint32_t j = i;
        while (j > 0u && local_peak_indices[j - 1u] > key)
        {
            local_peak_indices[j] = local_peak_indices[j - 1u];
            j -= 1u;
        }
        local_peak_indices[j] = key;
    }

    if (local_peak_count < 4u)
        return;

    bool has_best = false;
    float best_score = 0.0f;
    uint32_t best_corner_indices[4];
    QuadLineFitMoments best_edge_moments[4];

    for (uint32_t i0 = 0u; i0 + 3u < local_peak_count; ++i0)
    {
        for (uint32_t i1 = i0 + 1u; i1 + 2u < local_peak_count; ++i1)
        {
            for (uint32_t i2 = i1 + 1u; i2 + 1u < local_peak_count; ++i2)
            {
                for (uint32_t i3 = i2 + 1u; i3 < local_peak_count; ++i3)
                {
                    uint32_t corner_indices[4];
                    corner_indices[0] = local_peak_indices[i0];
                    corner_indices[1] = local_peak_indices[i1];
                    corner_indices[2] = local_peak_indices[i2];
                    corner_indices[3] = local_peak_indices[i3];

                    uint32_t edge_start[4];
                    uint32_t edge_end[4];
                    edge_start[0] = corner_indices[0];
                    edge_start[1] = corner_indices[1];
                    edge_start[2] = corner_indices[2];
                    edge_start[3] = corner_indices[3];
                    edge_end[0] = corner_indices[1];
                    edge_end[1] = corner_indices[2];
                    edge_end[2] = corner_indices[3];
                    edge_end[3] = corner_indices[0];

                    bool valid = true;
                    float total_mse = 0.0f;
                    QuadLineFitMoments edge_moments[4];

                    for (uint32_t edge_index = 0u; edge_index < 4u; ++edge_index)
                    {
                        QuadLineFitMoments moments;
                        segment_moments(
                            line_fit_points,
                            blob_start,
                            blob_count,
                            edge_start[edge_index],
                            edge_end[edge_index],
                            moments);

                        if (moments.N < 2)
                        {
                            valid = false;
                            break;
                        }

                        float mse = line_fit_mse(moments);
                        if (!(mse == mse) || mse > max_line_fit_mse)
                        {
                            valid = false;
                            break;
                        }

                        edge_moments[edge_index] = moments;
                        total_mse += mse;
                    }

                    if (!valid)
                        continue;

                    if (!has_best || total_mse < best_score)
                    {
                        has_best = true;
                        best_score = total_mse;
                        for (uint32_t c = 0u; c < 4u; ++c)
                        {
                            best_corner_indices[c] = corner_indices[c];
                            best_edge_moments[c] = edge_moments[c];
                        }
                    }
                }
            }
        }
    }

    if (!has_best)
        return;

    float4 lines[4];
    bool valid_lines = true;
    for (uint32_t line_index = 0u; line_index < 4u; ++line_index)
    {
        if (!fit_line_from_moments(best_edge_moments[line_index], lines[line_index]))
        {
            valid_lines = false;
            break;
        }
    }
    if (!valid_lines)
        return;

    bool bad_determinant = false;
    float2 corners[4];
    for (uint32_t i = 0u; i < 4u; ++i)
    {
        uint32_t i_next = (i + 1u) & 3u;
        float A00 = lines[i].w;
        float A01 = -lines[i_next].w;
        float A10 = -lines[i].z;
        float A11 = lines[i_next].z;
        float B0 = -lines[i].x + lines[i_next].x;
        float B1 = -lines[i].y + lines[i_next].y;

        float det = A00 * A11 - A10 * A01;
        if (abs(det) < 0.001f)
        {
            bad_determinant = true;
            break;
        }

        float W00 = A11 / det;
        float W01 = -A01 / det;
        float L0 = W00 * B0 + W01 * B1;
        corners[i].x = lines[i].x + L0 * A00;
        corners[i].y = lines[i].y + L0 * A10;
    }
    if (bad_determinant)
        return;

    float area = triangle_area(corners[0], corners[1], corners[2])
        + triangle_area(corners[2], corners[3], corners[0]);
    float min_area = 0.95f * float(min_tag_width * min_tag_width);
    if (area < min_area)
        return;

    bool reject_corner = false;
    for (uint32_t i = 0u; i < 4u; ++i)
    {
        uint32_t i0 = i;
        uint32_t i1 = (i + 1u) & 3u;
        uint32_t i2 = (i + 2u) & 3u;

        float dx1 = corners[i1].x - corners[i0].x;
        float dy1 = corners[i1].y - corners[i0].y;
        float dx2 = corners[i2].x - corners[i1].x;
        float dy2 = corners[i2].y - corners[i1].y;
        float denom = sqrt(dx1 * dx1 + dy1 * dy1) * sqrt(dx2 * dx2 + dy2 * dy2);
        if (denom < 1.0e-9f)
        {
            reject_corner = true;
            break;
        }

        float cos_dtheta = (dx1 * dx2 + dy1 * dy2) / denom;
        if (abs(cos_dtheta) > cos_critical_rad || dx1 * dy2 < dy1 * dx2)
        {
            reject_corner = true;
            break;
        }
    }
    if (reject_corner)
        return;

    if (quad_decimate > 1.0f)
    {
        if (abs(quad_decimate - 1.5f) < 1.0e-4f)
        {
            [ForceUnroll]
            for (uint32_t c = 0u; c < 4u; ++c)
            {
                corners[c] *= quad_decimate;
            }
        }
        else
        {
            [ForceUnroll]
            for (uint32_t c = 0u; c < 4u; ++c)
            {
                corners[c] = (corners[c] - 0.5f) * quad_decimate + 0.5f;
            }
        }
    }

    uint32_t fitted_quad_index;
    InterlockedAdd(fitted_quad_count_out[0], 1u, fitted_quad_index);
    write_fitted_quad_word(fitted_quads_out, fitted_quad_index, kFittedQuadBlobIndexWord, blob_index);
    write_fitted_quad_word(
        fitted_quads_out,
        fitted_quad_index,
        kFittedQuadReversedBorderWord,
        blob_extent_dot(filtered_blob_extents, blob_index) < 0.0f ? 1u : 0u);
    write_fitted_quad_word(fitted_quads_out, fitted_quad_index, kFittedQuadScoreWord, asuint(best_score));
    write_fitted_quad_word(fitted_quads_out, fitted_quad_index, kFittedQuadCorner0XWord, asuint(corners[0].x));
    write_fitted_quad_word(fitted_quads_out, fitted_quad_index, kFittedQuadCorner0YWord, asuint(corners[0].y));
    write_fitted_quad_word(fitted_quads_out, fitted_quad_index, kFittedQuadCorner1XWord, asuint(corners[1].x));
    write_fitted_quad_word(fitted_quads_out, fitted_quad_index, kFittedQuadCorner1YWord, asuint(corners[1].y));
    write_fitted_quad_word(fitted_quads_out, fitted_quad_index, kFittedQuadCorner2XWord, asuint(corners[2].x));
    write_fitted_quad_word(fitted_quads_out, fitted_quad_index, kFittedQuadCorner2YWord, asuint(corners[2].y));
    write_fitted_quad_word(fitted_quads_out, fitted_quad_index, kFittedQuadCorner3XWord, asuint(corners[3].x));
    write_fitted_quad_word(fitted_quads_out, fitted_quad_index, kFittedQuadCorner3YWord, asuint(corners[3].y));
    write_fitted_quad_word(
        fitted_quads_out,
        fitted_quad_index,
        kFittedQuadCornerIndex0Word,
        blob_start + best_corner_indices[0]);
    write_fitted_quad_word(
        fitted_quads_out,
        fitted_quad_index,
        kFittedQuadCornerIndex1Word,
        blob_start + best_corner_indices[1]);
    write_fitted_quad_word(
        fitted_quads_out,
        fitted_quad_index,
        kFittedQuadCornerIndex2Word,
        blob_start + best_corner_indices[2]);
    write_fitted_quad_word(
        fitted_quads_out,
        fitted_quad_index,
        kFittedQuadCornerIndex3Word,
        blob_start + best_corner_indices[3]);
}
