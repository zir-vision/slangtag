import "./image_buffer.slang";
import "../ccl.slang";

RWStructuredBuffer<uint32_t> labels;

struct PushConstants
{
    ImageSize image_size;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint2 tid = dispatch_thread_id.xy;
    uint2 dim = uint2(pc.image_size.width, pc.image_size.height);
    tid *= 2;
    if (any(tid >= dim))
        return;

    uint32_t idx = tid.y * dim.x + tid.x;
    uint32_t background_labels_index = (tid.y + 1) * dim.x + tid.x;
    find_and_compress(labels, idx);
    find_and_compress(labels, background_labels_index);
    find_and_compress(labels, background_labels_index + 1);
}
