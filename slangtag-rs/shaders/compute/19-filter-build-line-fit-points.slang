import "./image_buffer.slang";
import "../filter.slang";

StructuredBuffer<uint32_t> sorted_selected_points;
StructuredBuffer<uint32_t> decimated_image;
RWStructuredBuffer<uint32_t> line_fit_points_out;

struct PushConstants
{
    ImageSize decimated_size;
    uint32_t point_count;
    uint32_t decimate;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void build_line_fit_points(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    build_line_fit_points_buffer(
        dispatch_thread_id.x,
        sorted_selected_points,
        decimated_image,
        pc.decimated_size,
        line_fit_points_out,
        pc.point_count,
        pc.decimate);
}
