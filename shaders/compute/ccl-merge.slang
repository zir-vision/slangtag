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

    uint32_t foreground_labels_index = tid.y * dim.x + tid.x;
    uint32_t background_labels_index = (tid.y + 1) * dim.x + tid.x;
    uint32_t info = labels[foreground_labels_index + 1];

    uint8_t info_foreground = uint8_t(info & 0xff);
    uint8_t info_left_background = uint8_t((info >> 8) & 0xff);
    uint8_t info_right_background = uint8_t((info >> 16) & 0xff);

    if (has_bit(info_foreground, Info::Q))
        union_sets(labels, foreground_labels_index, foreground_labels_index - 2 * dim.x);
    if (has_bit(info_foreground, Info::R))
        union_sets(labels, foreground_labels_index, foreground_labels_index - 2 * dim.x + 2);
    if (has_bit(info_foreground, Info::S))
        union_sets(labels, foreground_labels_index, foreground_labels_index - 2);
    if (has_bit(info_left_background, Info::S))
        union_sets(labels, background_labels_index, background_labels_index - 1);
    if (has_bit(info_right_background, Info::S))
        union_sets(labels, background_labels_index + 1, background_labels_index);
}
