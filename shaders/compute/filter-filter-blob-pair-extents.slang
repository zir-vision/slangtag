import "../filter.slang";

StructuredBuffer<uint32_t> extents_in;
RWStructuredBuffer<uint32_t> extents_out;
RWStructuredBuffer<uint32_t> selected_extent_count_out;
RWStructuredBuffer<uint32_t> selected_point_count_out;
struct PushConstants
{
    uint32_t extent_count;
    uint32_t tag_width;
    uint32_t reversed_border;
    uint32_t normal_border;
    uint32_t min_cluster_pixels;
    uint32_t max_cluster_pixels;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint32_t tid = dispatch_thread_id.x;
    uint32_t extent_count = pc.extent_count;
    uint32_t tag_width = pc.tag_width;
    uint32_t reversed_border = pc.reversed_border;
    uint32_t normal_border = pc.normal_border;
    uint32_t min_cluster_pixels = pc.min_cluster_pixels;
    uint32_t max_cluster_pixels = pc.max_cluster_pixels;
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
