import "../filter.slang";
import "./detect-control.slang";

StructuredBuffer<uint32_t> line_fit_points;
StructuredBuffer<uint32_t> filtered_extents;
RWStructuredBuffer<uint32_t> errs_out;
RWStructuredBuffer<uint32_t> filtered_errs_out;
RWStructuredBuffer<uint32_t> peaks_out;
StructuredBuffer<uint32_t> control;
struct PushConstants
{
    uint32_t extent_count;
    uint32_t point_count;
    uint32_t pass;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    static const uint32_t kInvalidBlobIndex = 0xFFFFu;
    static const int kFilterRadius = 3;
    static const int kFilterTaps = 7;

    uint32_t tid = dispatch_thread_id.x;
    uint32_t extent_count = control[kControlBlobExtentCountWord];
    uint32_t point_count = control[kControlSelectedBlobPointCountWord];
    uint32_t pass = pc.pass;
    if (tid >= point_count)
        return;

    if (pass == 0u)
    {
        errs_out[tid] = asuint(0.0f);
        filtered_errs_out[tid] = asuint(0.0f);
        PeakPoint peak;
        peak.blob_index = kInvalidBlobIndex;
        peak.error = asuint(0.0f);
        peak.point_index = tid;
        write_peak(peaks_out, tid, peak);

        LineFitPoint line_fit = read_line_fit_point(line_fit_points, tid);
        uint32_t blob_index = line_fit.blob_index;

        if (blob_index >= extent_count)
            return;

        BlobExtent extent = read_blob_extent(filtered_extents, blob_index);
        if (extent.count == 0u || tid < extent.start || tid >= extent.start + extent.count)
            return;

        uint32_t local_index = tid - extent.start;
        uint32_t ksz = min(20u, extent.count / 12u);
        uint32_t i0 = (local_index + 2u * extent.count - ksz) % extent.count;
        uint32_t i1 = (local_index + extent.count + ksz) % extent.count;

        LineFitPoint window;
        int N;

        if (i0 < i1)
        {
            N = int(i1 - i0 + 1u);
            window = read_line_fit_point(line_fit_points, extent.start + i1);

            if (i0 > 0u)
            {
                LineFitPoint prefix = read_line_fit_point(line_fit_points, extent.start + i0 - 1u);
                window.Mx -= prefix.Mx;
                window.My -= prefix.My;
                window.W -= prefix.W;
                window.Mxx -= prefix.Mxx;
                window.Myy -= prefix.Myy;
                window.Mxy -= prefix.Mxy;
            }
        }
        else
        {
            LineFitPoint prefix = read_line_fit_point(line_fit_points, extent.start + i0 - 1u);
            LineFitPoint tail = read_line_fit_point(line_fit_points, extent.start + extent.count - 1u);
            LineFitPoint head = read_line_fit_point(line_fit_points, extent.start + i1);

            window.Mx = (tail.Mx - prefix.Mx) + head.Mx;
            window.My = (tail.My - prefix.My) + head.My;
            window.W = (tail.W - prefix.W) + head.W;
            window.Mxx = (tail.Mxx - prefix.Mxx) + head.Mxx;
            window.Myy = (tail.Myy - prefix.Myy) + head.Myy;
            window.Mxy = (tail.Mxy - prefix.Mxy) + head.Mxy;
            N = int(extent.count - i0 + i1 + 1u);
        }

        float err = fit_line_error(N, window.Mx, window.My, window.Mxx, window.Myy, window.Mxy, window.W);
        errs_out[tid] = asuint(err);
        return;
    }

    if (pass == 1u)
    {
        uint32_t blob_index = read_line_fit_point(line_fit_points, tid).blob_index;

        if (blob_index >= extent_count)
            return;

        BlobExtent extent = read_blob_extent(filtered_extents, blob_index);
        if (extent.count == 0u || tid < extent.start || tid >= extent.start + extent.count)
            return;

        int local_index = int(tid - extent.start);
        float filtered_err = 0.0f;
        for (int j = 0; j < kFilterTaps; ++j)
        {
            int neighbor_local_index = (local_index + int(extent.count) + j - kFilterRadius) % int(extent.count);
            uint32_t neighbor_index = extent.start + uint32_t(neighbor_local_index);
            filtered_err += asfloat(errs_out[neighbor_index]) * line_fit_filter_coefficient(j);
        }

        filtered_errs_out[tid] = asuint(filtered_err);
        return;
    }

    if (pass == 2u)
    {
        uint32_t blob_index = read_line_fit_point(line_fit_points, tid).blob_index;

        if (blob_index >= extent_count)
            return;

        BlobExtent extent = read_blob_extent(filtered_extents, blob_index);
        if (extent.count == 0u || tid < extent.start || tid >= extent.start + extent.count)
            return;

        uint32_t local_index = tid - extent.start;
        uint32_t before_index = extent.start + ((local_index + extent.count - 1u) % extent.count);
        uint32_t after_index = extent.start + ((local_index + 1u) % extent.count);

        float before_error = asfloat(filtered_errs_out[before_index]);
        float my_error = asfloat(filtered_errs_out[tid]);
        float after_error = asfloat(filtered_errs_out[after_index]);
        bool is_peak = my_error > before_error && my_error > after_error;

        PeakPoint peak;
        peak.blob_index = is_peak ? blob_index : kInvalidBlobIndex;
        peak.error = asuint(-my_error);
        peak.point_index = tid;
        write_peak(peaks_out, tid, peak);
    }
}
