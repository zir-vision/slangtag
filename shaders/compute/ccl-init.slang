import "./image_buffer.slang";
import "../ccl.slang";

StructuredBuffer<uint8_t> image;
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
    tid *= 2;
    uint2 dim = uint2(pc.image_size.width, pc.image_size.height);
    
    if (any(tid >= dim))
        return;

    // Linear index calculation
    uint32_t idx = tid.y * dim.x + tid.x;

    uint32_t p_foreground = 0;
    uint32_t p_background = 0;

    uint8_t info_foreground = 0;
    uint8_t info_left_background = 0;
    uint8_t info_right_background = 0;
    
    // Keep values unsigned to avoid backend-dependent sign behavior.
    uint32_t buffer[4] = { 0u, 0u, 0u, 0u };

    buffer[0] = image_u8_read(image, pc.image_size, tid);
    buffer[1] = image_u8_read(image, pc.image_size, tid + uint2(1, 0));
    buffer[2] = image_u8_read(image, pc.image_size, tid + uint2(0, 1));
    buffer[3] = image_u8_read(image, pc.image_size, tid + uint2(1, 1));

    if (is_white_sample_u32(buffer[0]))
    {
        p_foreground |= 0x777;
        set_bit(info_foreground, Info::a);
    }
    else if (buffer[0] == 0)
    {
        p_background |= 0x272;
        set_bit(info_left_background, Info::a);
    }

    if (is_white_sample_u32(buffer[1]))
    {
        p_foreground |= (0x777 << 1);
        set_bit(info_foreground, Info::b);
    }
    else if (buffer[1] == 0)
    {
        p_background |= (0x272 << 1);
        set_bit(info_right_background, Info::b);
    }

    if (is_white_sample_u32(buffer[2]))
    {
        p_foreground |= (0x777 << 4);
        set_bit(info_foreground, Info::c);
    }
    else if (buffer[2] == 0)
    {
        p_background |= (0x272 << 4);
        set_bit(info_left_background, Info::c);
    }

    if (is_white_sample_u32(buffer[3]))
    {
        set_bit(info_foreground, Info::d);
    }
    else if (buffer[3] == 0)
    {
        set_bit(info_right_background, Info::d);
    }

    if (tid.x == 0)
    {
        p_foreground &= 0xEEEE;
        p_background &= 0xEEEE;
    }

    if (tid.x + 2 >= dim.x)
    {
        p_foreground &= 0x7777;
        p_background &= 0x7777;
    }

    if (tid.y == 0)
    {
        p_foreground &= 0xFFF0;
        p_background &= 0xFFF0;
    }

    if (tid.y + 2 >= dim.y)
    {
        p_foreground &= 0x0FFF;
        p_background &= 0x0FFF;
    }

    int father_offset_foreground = 0;
    int father_offset_left_background = 0;
    int father_offset_right_background = 0;

    // --- Neighbor Checks ---
    // Note: neighbor lookups must ensure they don't go OOB. 
    // The original code relies on padding or careful bounds. 
    // Assuming 'image' handles OOB clamps or logic is sufficient here.

    // P Square
    if (tid.x > 0u &&
        tid.y > 0u &&
        has_bit(p_foreground, 0) &&
        is_white_sample_u32(image_u8_read(image, pc.image_size, tid - uint2(1, 1))))
    {
        father_offset_foreground = -(2 * int(dim.x) + 2);
    }

    // Q Square
    bool q_foreground =
        (tid.y > 0u &&
         has_bit(p_foreground, 1) &&
         is_white_sample_u32(image_u8_read(image, pc.image_size, tid - uint2(0, 1)))) ||
        (tid.y > 0u &&
         tid.x + 1u < dim.x &&
         has_bit(p_foreground, 2) &&
         is_white_sample_u32(image_u8_read(image, pc.image_size, tid + uint2(1, 0) - uint2(0, 1))));
    if (q_foreground)
    {
        if (father_offset_foreground == 0)
        {
            father_offset_foreground = -(2 * int(dim.x));
        }
        else
        {
            set_bit(info_foreground, Info::Q);
        }
    }

    if (tid.y > 0u &&
        has_bit(p_background, 1) &&
        image_u8_read(image, pc.image_size, tid - uint2(0, 1)) == 0u)
    {
        father_offset_left_background = -2 * int(dim.x);
    }
    if (tid.y > 0u &&
        tid.x + 1u < dim.x &&
        has_bit(p_background, 2) &&
        image_u8_read(image, pc.image_size, tid + uint2(1, 0) - uint2(0, 1)) == 0u)
    {
        father_offset_right_background = -2 * int(dim.x);
    }

    // R Square
    if (tid.y > 0u &&
        tid.x + 2u < dim.x &&
        has_bit(p_foreground, 3) &&
        is_white_sample_u32(image_u8_read(image, pc.image_size, tid + uint2(2, 0) - uint2(0, 1))))
    {
        if (father_offset_foreground == 0)
        {
            father_offset_foreground = -(2 * int(dim.x) - 2);
        }
        else
        {
            set_bit(info_foreground, Info::R);
        }
    }

    // S Square
    bool s_foreground =
        (tid.x > 0u &&
         has_bit(p_foreground, 4) &&
         is_white_sample_u32(image_u8_read(image, pc.image_size, tid - uint2(1, 0)))) ||
        (tid.x > 0u &&
         tid.y + 1u < dim.y &&
         has_bit(p_foreground, 8) &&
         is_white_sample_u32(image_u8_read(image, pc.image_size, tid + uint2(0, 1) - uint2(1, 0))));
    if (s_foreground)
    {
        if (father_offset_foreground == 0)
        {
            father_offset_foreground = -2;
        }
        else
        {
            set_bit(info_foreground, Info::S);
        }
    }
    bool s_background =
        (tid.x > 0u &&
         has_bit(p_background, 4) &&
         image_u8_read(image, pc.image_size, tid - uint2(1, 0)) == 0u) ||
        (tid.x > 0u &&
         tid.y + 1u < dim.y &&
         has_bit(p_background, 8) &&
         image_u8_read(image, pc.image_size, tid + uint2(0, 1) - uint2(1, 0)) == 0u);
    if (s_background)
    {
        if (father_offset_left_background == 0)
        {
            father_offset_left_background = -1;
        }
        else
        {
            set_bit(info_left_background, Info::S);
        }
    }

    if ((has_bit(info_left_background, Info::a) &&
         has_bit(info_right_background, Info::b)) ||
        (has_bit(info_left_background, Info::c) &&
         has_bit(info_right_background, Info::d)))
    {
        if (father_offset_right_background == 0)
        {
            father_offset_right_background = -1;
        }
        else
        {
            set_bit(info_right_background, Info::S);
        }
    }

    // FIX: Explicitly writing to linear buffer indices instead of Texture2D
    // Even cols (x, x) = Pointer, Odd cols (x+1, x+1) = Info/Pointer
    
    // Top-Left (Foreground Pointer)
    labels[idx] = uint32_t(int(idx) + father_offset_foreground);
    
    // Top-Right (Info Pack)
    labels[idx + 1] = uint32_t(info_foreground) +
                      (uint32_t(info_left_background) << 8) +
                      (uint32_t(info_right_background) << 16);
    
    uint32_t background_labels_index = idx + dim.x;

    // Bottom-Left (Left BG Pointer) - Next Row
    labels[background_labels_index] =
        uint32_t(int(background_labels_index) + father_offset_left_background);
    
    // Bottom-Right (Right BG Pointer) - Next Row
    labels[background_labels_index + 1] =
        uint32_t(int(background_labels_index) + father_offset_right_background + 1);
}
