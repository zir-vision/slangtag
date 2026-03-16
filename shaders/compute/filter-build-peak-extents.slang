import "../filter.slang";

StructuredBuffer<uint32_t> sorted_peaks;
RWStructuredBuffer<uint32_t> peak_extents_out;
RWStructuredBuffer<uint32_t> peak_extent_count_out;
struct PushConstants
{
    uint32_t valid_peaks;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint32_t tid = dispatch_thread_id.x;
    uint32_t valid_peaks = pc.valid_peaks;
    if (tid >= valid_peaks)
        return;

    uint32_t blob_index = read_peak_word(sorted_peaks, tid, kPeakBlobIndexWord);
    if (blob_index == 0xFFFFu)
        return;

    if (tid > 0u)
    {
        uint32_t previous_blob_index = read_peak_word(sorted_peaks, tid - 1u, kPeakBlobIndexWord);
        if (previous_blob_index == blob_index)
            return;
    }

    uint32_t start = tid;
    uint32_t count = 1u;
    while (start + count < valid_peaks &&
           read_peak_word(sorted_peaks, start + count, kPeakBlobIndexWord) == blob_index)
    {
        count += 1u;
    }

    uint32_t peak_extent_index;
    InterlockedAdd(peak_extent_count_out[0], 1u, peak_extent_index);
    write_peak_extent_word(peak_extents_out, peak_extent_index, kPeakExtentBlobIndexWord, blob_index);
    write_peak_extent_word(peak_extents_out, peak_extent_index, kPeakExtentStartingOffsetWord, start);
    write_peak_extent_word(peak_extents_out, peak_extent_index, kPeakExtentCountWord, count);
}
