import "../filter.slang";

StructuredBuffer<uint32_t> sorted_selected_points;
RWStructuredBuffer<uint32_t> filtered_extents;

struct PushConstants
{
    uint32_t extent_count;
    uint32_t point_count;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint32_t tid = dispatch_thread_id.x;
    if (tid >= pc.point_count)
        return;

    uint32_t blob_index =
        read_selected_blob_point_word(sorted_selected_points, tid, kSelectedPointBlobIndexWord);
    if (blob_index >= pc.extent_count)
        return;

    if (tid > 0u)
    {
        uint32_t previous_blob_index = read_selected_blob_point_word(
            sorted_selected_points, tid - 1u, kSelectedPointBlobIndexWord);
        if (previous_blob_index == blob_index)
            return;
    }

    if (read_extent_word(filtered_extents, blob_index, kExtentCountWord) == 0u)
        return;

    write_extent_word(filtered_extents, blob_index, kExtentStartingOffsetWord, tid);
}
