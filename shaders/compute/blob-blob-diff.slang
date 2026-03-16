import "./image_buffer.slang";
import "../blob.slang";

StructuredBuffer<uint8_t> thresholded_image;
RWStructuredBuffer<uint32_t> blobs;
RWStructuredBuffer<uint32_t> union_markers_size;
RWStructuredBuffer<uint32_t> result;

struct PushConstants
{
    ImageSize thresholded_size;
    uint32_t min_blob_size;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint32_t x = dispatch_thread_id.x;
    uint32_t y = dispatch_thread_id.y;
    uint32_t width = pc.thresholded_size.width;
    uint32_t height = pc.thresholded_size.height;

    if (x >= width || y >= height || y == 0u || x == 0u || x == width - 1u || y == height - 1u)
        return;

    uint32_t global_input_index = x + y * width;
    uint32_t global_output_index = (x - 1u) + (y - 1u) * (width - 2u);

    uint32_t rep0 = blobs[global_input_index];
    uint32_t v0 = image_u8_read(thresholded_image, pc.thresholded_size, uint2(x, y));

    if (v0 == 127u || union_markers_size[rep0] < pc.min_blob_size)
    {
        clear_blob_diff_point(result, width, height, 0u, global_output_index);
        clear_blob_diff_point(result, width, height, 1u, global_output_index);
        clear_blob_diff_point(result, width, height, 2u, global_output_index);
        clear_blob_diff_point(result, width, height, 3u, global_output_index);
        return;
    }

    uint32_t rep1_0 = blobs[global_input_index + 1u];
    uint32_t v1_0 = image_u8_read(thresholded_image, pc.thresholded_size, uint2(x + 1u, y));
    write_blob_diff_connection(
        result, union_markers_size, width, height, global_output_index, x, y, 0u, v0, v1_0, rep0, rep1_0, pc.min_blob_size);

    uint32_t rep1_1 = blobs[global_input_index + width + 1u];
    uint32_t v1_1 = image_u8_read(thresholded_image, pc.thresholded_size, uint2(x + 1u, y + 1u));
    write_blob_diff_connection(
        result, union_markers_size, width, height, global_output_index, x, y, 1u, v0, v1_1, rep0, rep1_1, pc.min_blob_size);

    uint32_t rep1_2 = blobs[global_input_index + width];
    uint32_t v1_2 = image_u8_read(thresholded_image, pc.thresholded_size, uint2(x, y + 1u));
    write_blob_diff_connection(
        result, union_markers_size, width, height, global_output_index, x, y, 2u, v0, v1_2, rep0, rep1_2, pc.min_blob_size);

    uint32_t rep_block_left = blobs[global_input_index - 1u];
    uint32_t v1_block_left = image_u8_read(thresholded_image, pc.thresholded_size, uint2(x - 1u, y));
    if (v1_block_left != 127u && v1_2 != 127u && v1_2 != v1_block_left)
    {
        if (x != 1u &&
            union_markers_size[rep_block_left] >= pc.min_blob_size &&
            union_markers_size[rep1_2] >= pc.min_blob_size)
        {
            clear_blob_diff_point(result, width, height, 3u, global_output_index);
            return;
        }
    }

    uint32_t rep1_3 = blobs[global_input_index + width - 1u];
    uint32_t v1_3 = image_u8_read(thresholded_image, pc.thresholded_size, uint2(x - 1u, y + 1u));
    write_blob_diff_connection(
        result, union_markers_size, width, height, global_output_index, x, y, 3u, v0, v1_3, rep0, rep1_3, pc.min_blob_size);
}
