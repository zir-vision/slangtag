import "./image_buffer.slang";
import "../ccl.slang";

RWStructuredBuffer<uint32_t> labels;
RWStructuredBuffer<uint32_t> final_labels;
RWStructuredBuffer<uint32_t> union_markers_size_device;

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
    uint32_t foreground_label = find(labels, labels[foreground_labels_index]);
    uint32_t foreground_info_packed = labels[foreground_labels_index + 1];
    uint32_t background_left_label = find(labels, labels[background_labels_index]);
    uint32_t background_right_label = find(labels, labels[background_labels_index + 1]);
    uint8_t foreground_info = uint8_t(foreground_info_packed & 0xFF);
    uint8_t background_left_info = uint8_t((foreground_info_packed >> 8) & 0xFF);
    uint8_t background_right_info = uint8_t((foreground_info_packed >> 16) & 0xFF);

    uint32_t a_label;
    uint32_t b_label;
    uint32_t c_label;
    uint32_t d_label;

    if ((foreground_info & 0xf) == 0u && (background_left_info & 0xf) == 0u &&
        (background_right_info & 0xf) == 0u)
    {
        a_label = foreground_labels_index;
        b_label = foreground_labels_index + 1;
        c_label = background_labels_index;
        d_label = background_labels_index + 1;
        final_labels[foreground_labels_index] = a_label;
        final_labels[foreground_labels_index + 1] = b_label;
        final_labels[background_labels_index] = c_label;
        final_labels[background_labels_index + 1] = d_label;
        return;
    }
    else
    {
        a_label = (has_bit(foreground_info, Info::a) ? foreground_label : 0u) +
                  (has_bit(background_left_info, Info::a) ? background_left_label : 0u);
        b_label = (has_bit(foreground_info, Info::b) ? foreground_label : 0u) +
                  (has_bit(background_right_info, Info::b) ? background_right_label : 0u);
        c_label = (has_bit(foreground_info, Info::c) ? foreground_label : 0u) +
                  (has_bit(background_left_info, Info::c) ? background_left_label : 0u);
        d_label = (has_bit(foreground_info, Info::d) ? foreground_label : 0u) +
                  (has_bit(background_right_info, Info::d) ? background_right_label : 0u);
        final_labels[foreground_labels_index] = a_label;
        final_labels[foreground_labels_index + 1] = b_label;
        final_labels[background_labels_index] = c_label;
        final_labels[background_labels_index + 1] = d_label;
    }

    if ((foreground_info & 0xf) != 0u)
    {
        uint count = 0;
        if (has_bit(foreground_info, Info::a)) ++count;
        if (has_bit(foreground_info, Info::b)) ++count;
        if (has_bit(foreground_info, Info::c)) ++count;
        if (has_bit(foreground_info, Info::d)) ++count;
        InterlockedAdd(union_markers_size_device[foreground_label], count);
    }

    if ((background_left_info & 0xf) == 0u && (background_right_info & 0xf) == 0u)
        return;

    if ((background_left_info & 0xf) != 0u &&
        (background_right_info & 0xf) != 0u &&
        background_left_label == background_right_label)
    {
        uint count = 0;
        if (has_bit(background_left_info, Info::a)) ++count;
        if (has_bit(background_right_info, Info::b)) ++count;
        if (has_bit(background_left_info, Info::c)) ++count;
        if (has_bit(background_right_info, Info::d)) ++count;
        InterlockedAdd(union_markers_size_device[background_left_label], count);
        return;
    }

    if ((background_left_info & 0xf) != 0u)
    {
        uint count = 0;
        if (has_bit(background_left_info, Info::a)) ++count;
        if (has_bit(background_left_info, Info::c)) ++count;
        InterlockedAdd(union_markers_size_device[background_left_label], count);
    }

    if ((background_right_info & 0xf) != 0u)
    {
        uint count = 0;
        if (has_bit(background_right_info, Info::b)) ++count;
        if (has_bit(background_right_info, Info::d)) ++count;
        InterlockedAdd(union_markers_size_device[background_right_label], count);
    }
}
