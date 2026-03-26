import "./detect-control.slang";

StructuredBuffer<uint32_t> control_in;
RWStructuredBuffer<uint32_t> control_out;

struct PushConstants
{
    uint32_t one_d_local_size_x;
    uint32_t marker_size_with_borders;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    if (dispatch_thread_id.x != 0u || dispatch_thread_id.y != 0u || dispatch_thread_id.z != 0u)
        return;

    uint32_t one_d_local_size_x = max(pc.one_d_local_size_x, 1u);
    uint32_t blob_diff_filtered_count = control_in[kControlBlobDiffFilteredCountWord];
    uint32_t blob_extent_count = control_in[kControlBlobExtentCountWord];
    uint32_t selected_blob_point_count = control_in[kControlSelectedBlobPointCountWord];
    uint32_t peak_extent_count = control_in[kControlPeakExtentCountWord];
    uint32_t fitted_quad_count = control_in[kControlFittedQuadCountWord];

    write_dispatch_1d(
        control_out,
        kControlDispatchBuildBlobPairExtentsWord,
        blob_diff_filtered_count,
        one_d_local_size_x);
    write_dispatch_1d(
        control_out,
        kControlDispatchRadixBlobInitWord,
        blob_diff_filtered_count,
        one_d_local_size_x);
    write_dispatch_1d(
        control_out,
        kControlDispatchRadixBlobUpdateWord,
        blob_diff_filtered_count,
        one_d_local_size_x);
    write_dispatch_1d(
        control_out,
        kControlDispatchRadixBlobGatherWord,
        blob_diff_filtered_count,
        one_d_local_size_x);

    write_dispatch_1d(
        control_out,
        kControlDispatchRadixSelectedInitWord,
        selected_blob_point_count,
        one_d_local_size_x);
    write_dispatch_1d(
        control_out,
        kControlDispatchRadixSelectedUpdateWord,
        selected_blob_point_count,
        one_d_local_size_x);
    write_dispatch_1d(
        control_out,
        kControlDispatchRadixSelectedGatherWord,
        selected_blob_point_count,
        one_d_local_size_x);
    write_dispatch_1d(
        control_out,
        kControlDispatchRebuildFilteredExtentStartsWord,
        selected_blob_point_count,
        one_d_local_size_x);
    write_dispatch_1d(
        control_out,
        kControlDispatchBuildLineFitPointsWord,
        blob_extent_count,
        one_d_local_size_x);
    write_dispatch_1d(
        control_out,
        kControlDispatchFitLinePass0Word,
        selected_blob_point_count,
        one_d_local_size_x);
    write_dispatch_1d(
        control_out,
        kControlDispatchFitLinePass1Word,
        selected_blob_point_count,
        one_d_local_size_x);
    write_dispatch_1d(
        control_out,
        kControlDispatchFitLinePass2Word,
        selected_blob_point_count,
        one_d_local_size_x);

    write_dispatch_1d(
        control_out,
        kControlDispatchRadixPeaksInitWord,
        selected_blob_point_count,
        one_d_local_size_x);
    write_dispatch_1d(
        control_out,
        kControlDispatchRadixPeaksUpdateWord,
        selected_blob_point_count,
        one_d_local_size_x);
    write_dispatch_1d(
        control_out,
        kControlDispatchRadixPeaksGatherWord,
        selected_blob_point_count,
        one_d_local_size_x);
    write_dispatch_1d(
        control_out,
        kControlDispatchBuildPeakExtentsWord,
        selected_blob_point_count,
        one_d_local_size_x);

    write_dispatch_1d(
        control_out,
        kControlDispatchFitQuadsWord,
        peak_extent_count,
        one_d_local_size_x);
    write_dispatch_1d(
        control_out,
        kControlDispatchPrepareDecodeQuadsWord,
        fitted_quad_count,
        one_d_local_size_x);

    uint32_t cells_per_quad = pc.marker_size_with_borders * pc.marker_size_with_borders;
    uint32_t total_cells = 0u;
    if (cells_per_quad > 0u && fitted_quad_count <= (0xFFFFFFFFu / cells_per_quad))
        total_cells = fitted_quad_count * cells_per_quad;
    write_dispatch_1d(
        control_out,
        kControlDispatchExtractCandidateBitsWord,
        total_cells,
        one_d_local_size_x);
    write_dispatch_1d(
        control_out,
        kControlDispatchDecodeFinalizeWord,
        fitted_quad_count,
        one_d_local_size_x);
}
