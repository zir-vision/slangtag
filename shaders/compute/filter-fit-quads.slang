import "../filter.slang";

StructuredBuffer<uint32_t> sorted_peaks;
StructuredBuffer<uint32_t> peak_extents;
StructuredBuffer<uint32_t> line_fit_points;
StructuredBuffer<uint32_t> filtered_blob_extents;
RWStructuredBuffer<uint32_t> fitted_quads_out;
RWStructuredBuffer<uint32_t> fitted_quad_count_out;

struct PushConstants
{
    uint32_t peak_extent_count;
    uint32_t filtered_blob_extent_count;
    uint32_t max_nmaxima;
    float max_line_fit_mse;
    float cos_critical_rad;
    uint32_t min_tag_width;
    float quad_decimate;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

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

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint32_t tid = dispatch_thread_id.x;
    uint32_t peak_extent_count = pc.peak_extent_count;
    uint32_t filtered_blob_extent_count = pc.filtered_blob_extent_count;
    uint32_t max_nmaxima = pc.max_nmaxima;
    float max_line_fit_mse = pc.max_line_fit_mse;
    float cos_critical_rad = pc.cos_critical_rad;
    uint32_t min_tag_width = pc.min_tag_width;
    float quad_decimate = pc.quad_decimate;
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
