import "./image_buffer.slang";
import "../blob.slang";

StructuredBuffer<uint32_t> thresholded_image;
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
[numthreads(1, 1, 1)]
void blob_diff(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    blob_diff_buffer(
        dispatch_thread_id.xy,
        thresholded_image,
        pc.thresholded_size,
        blobs,
        union_markers_size,
        result,
        uint2(pc.thresholded_size.width, pc.thresholded_size.height),
        pc.min_blob_size);
}
