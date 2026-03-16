import "../filter.slang";

StructuredBuffer<uint32_t> line_fit_points;
StructuredBuffer<uint32_t> filtered_extents;
RWStructuredBuffer<uint32_t> errs_out;
RWStructuredBuffer<uint32_t> filtered_errs_out;
RWStructuredBuffer<uint32_t> peaks_out;
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
    uint32_t tid = dispatch_thread_id.x;
    uint32_t extent_count = pc.extent_count;
    uint32_t point_count = pc.point_count;
    uint32_t pass = pc.pass;
    if (tid >= point_count)
        return;

    if (pass == 0u)
    {
        errs_out[tid] = asuint(0.0f);
        filtered_errs_out[tid] = asuint(0.0f);
        write_peak_word(peaks_out, tid, kPeakBlobIndexWord, 0xFFFFu);
        write_peak_word(peaks_out, tid, kPeakErrorWord, asuint(0.0f));
        write_peak_word(peaks_out, tid, kPeakPointIndexWord, tid);

        int Mx_dummy;
        int My_dummy;
        int W_dummy;
        uint32_t blob_index;
        int64_t Mxx_dummy;
        int64_t Myy_dummy;
        int64_t Mxy_dummy;
        read_line_fit_point(line_fit_points, tid, Mx_dummy, My_dummy, W_dummy, blob_index, Mxx_dummy, Myy_dummy, Mxy_dummy);

        if (blob_index >= extent_count)
            return;

        uint32_t start = read_extent_word(filtered_extents, blob_index, kExtentStartingOffsetWord);
        uint32_t count = read_extent_word(filtered_extents, blob_index, kExtentCountWord);
        if (count == 0u || tid < start || tid >= start + count)
            return;

        uint32_t local_index = tid - start;
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
        errs_out[tid] = asuint(err);
        return;
    }

    if (pass == 1u)
    {
        int Mx_dummy;
        int My_dummy;
        int W_dummy;
        uint32_t blob_index;
        int64_t Mxx_dummy;
        int64_t Myy_dummy;
        int64_t Mxy_dummy;
        read_line_fit_point(line_fit_points, tid, Mx_dummy, My_dummy, W_dummy, blob_index, Mxx_dummy, Myy_dummy, Mxy_dummy);

        if (blob_index >= extent_count)
            return;

        uint32_t start = read_extent_word(filtered_extents, blob_index, kExtentStartingOffsetWord);
        uint32_t count = read_extent_word(filtered_extents, blob_index, kExtentCountWord);
        if (count == 0u || tid < start || tid >= start + count)
            return;

        int local_index = int(tid - start);
        float filtered_err = 0.0f;
        for (int j = 0; j < 7; ++j)
        {
            int neighbor_local_index = (local_index + int(count) + j - 3) % int(count);
            uint32_t neighbor_index = start + uint32_t(neighbor_local_index);
            filtered_err += asfloat(errs_out[neighbor_index]) * line_fit_filter_coefficient(j);
        }

        filtered_errs_out[tid] = asuint(filtered_err);
        return;
    }

    if (pass == 2u)
    {
        int Mx_dummy;
        int My_dummy;
        int W_dummy;
        uint32_t blob_index;
        int64_t Mxx_dummy;
        int64_t Myy_dummy;
        int64_t Mxy_dummy;
        read_line_fit_point(line_fit_points, tid, Mx_dummy, My_dummy, W_dummy, blob_index, Mxx_dummy, Myy_dummy, Mxy_dummy);

        if (blob_index >= extent_count)
            return;

        uint32_t start = read_extent_word(filtered_extents, blob_index, kExtentStartingOffsetWord);
        uint32_t count = read_extent_word(filtered_extents, blob_index, kExtentCountWord);
        if (count == 0u || tid < start || tid >= start + count)
            return;

        uint32_t local_index = tid - start;
        uint32_t before_index = start + ((local_index + count - 1u) % count);
        uint32_t after_index = start + ((local_index + 1u) % count);

        float before_error = asfloat(filtered_errs_out[before_index]);
        float my_error = asfloat(filtered_errs_out[tid]);
        float after_error = asfloat(filtered_errs_out[after_index]);
        bool is_peak = my_error > before_error && my_error > after_error;

        write_peak_word(peaks_out, tid, kPeakBlobIndexWord, is_peak ? blob_index : 0xFFFFu);
        write_peak_word(peaks_out, tid, kPeakErrorWord, asuint(-my_error));
        write_peak_word(peaks_out, tid, kPeakPointIndexWord, tid);
    }
}
