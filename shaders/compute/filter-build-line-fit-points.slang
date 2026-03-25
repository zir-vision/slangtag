import "./image_buffer.slang";
import "../filter.slang";
import "./detect-control.slang";

StructuredBuffer<uint32_t> sorted_selected_points;
StructuredBuffer<uint8_t> decimated_image;
StructuredBuffer<uint32_t> filtered_extents;
RWStructuredBuffer<uint32_t> line_fit_points_out;
StructuredBuffer<uint32_t> control;

struct PushConstants
{
    ImageSize decimated_size;
    uint32_t extent_count;
    uint32_t point_count;
    uint32_t decimate;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    static const int kOne = 1;

    uint tid = dispatch_thread_id.x;
    uint32_t extent_count = control[kControlBlobExtentCountWord];
    uint32_t point_count = control[kControlSelectedBlobPointCountWord];
    if (tid >= extent_count)
        return;

    uint32_t blob_index = tid;
    BlobExtent extent = read_blob_extent(filtered_extents, blob_index);
    if (extent.count == 0u)
        return;

    LineFitPoint running;
    running.Mx = 0;
    running.My = 0;
    running.W = 0;
    running.blob_index = blob_index;
    running.Mxx = 0;
    running.Myy = 0;
    running.Mxy = 0;

    for (uint32_t local_index = 0u; local_index < extent.count; ++local_index)
    {
        uint32_t i = extent.start + local_index;
        if (i >= point_count)
            return;

        SelectedBlobPoint point = read_selected_blob_point(sorted_selected_points, i);

        // Keep line-fit moments in fixed x2 coordinates (2*x+1) regardless of decimation.
        // Decimation-to-fullres scaling is handled later in fit-quads.
        int ix2 = int(point.x) * 2 + 1;
        int iy2 = int(point.y) * 2 + 1;
        int ix = ix2 / 2;
        int iy = iy2 / 2;

        int W = kOne;
        if (ix > 0 && uint32_t(ix + 1) < pc.decimated_size.width && iy > 0 && uint32_t(iy + 1) < pc.decimated_size.height)
        {
            int grad_x = int(image_u8_read(decimated_image, pc.decimated_size, uint2(uint32_t(ix + 1), uint32_t(iy))))
                - int(image_u8_read(decimated_image, pc.decimated_size, uint2(uint32_t(ix - 1), uint32_t(iy))));
            int grad_y = int(image_u8_read(decimated_image, pc.decimated_size, uint2(uint32_t(ix), uint32_t(iy + 1))))
                - int(image_u8_read(decimated_image, pc.decimated_size, uint2(uint32_t(ix), uint32_t(iy - 1))));

            W = int(sqrt(float(grad_x * grad_x + grad_y * grad_y))) + kOne;
        }

        running.Mx += W * ix2;
        running.My += W * iy2;
        running.W += W;
        running.Mxx += int64_t(W) * int64_t(ix2) * int64_t(ix2);
        running.Mxy += int64_t(W) * int64_t(ix2) * int64_t(iy2);
        running.Myy += int64_t(W) * int64_t(iy2) * int64_t(iy2);

        write_line_fit_point(line_fit_points_out, i, running);
    }
}
