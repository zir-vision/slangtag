import "compute/image_buffer.slang";

enum Info : uint8_t
{
    a = 0,
    b = 1,
    c = 2,
    d = 3,
    P = 4,
    Q = 5,
    R = 6,
    S = 7
};

bool has_bit(uint32_t bitmap, uint8_t position)
{
    return ((bitmap >> position) & 1u) != 0;
}

bool has_bit(uint32_t bitmap, Info position)
{
    return has_bit(bitmap, uint8_t(position));
}

void set_bit(inout uint8_t bitmap, Info position)
{
    bitmap |= (uint8_t(1) << uint8_t(position));
}

bool is_white_sample_u32(uint32_t v)
{
    return v != 0u && v != 127u;
}

// FIX: Use RWStructuredBuffer for linear addressing support
uint32_t find(RWStructuredBuffer<uint32_t> s_buf, uint32_t n)
{
    while (s_buf[n] != n)
    {
        n = s_buf[n];
    }
    return n;
}

uint32_t find_and_compress(RWStructuredBuffer<uint32_t> s_buf, uint32_t n)
{
    uint32_t id = n;
    while (s_buf[n] != n)
    {
        n = s_buf[n];
        s_buf[id] = n;
    }
    return n;
}

// FIX: Renamed from 'union' to 'union_sets' to avoid keyword collision
void union_sets(RWStructuredBuffer<uint32_t> s_buf, uint32_t a, uint32_t b)
{
    bool done = false;
    do
    {
        a = find(s_buf, a);
        b = find(s_buf, b);

        if (a < b)
        {
            uint32_t old;
            InterlockedMin(s_buf[b], a, old);
            done = (old == b);
            b = old;
        }
        else if (a > b)
        {
            uint32_t old;
            InterlockedMin(s_buf[a], b, old);
            done = (old == a);
            a = old;
        }
        else
        {
            done = true;
        }

    } while (!done);
}

// FIX: labels is now RWStructuredBuffer
void init(uint2 tid, RWTexture2D<uint8_t1> image, RWStructuredBuffer<uint32_t> labels)
{
    tid *= 2;
    uint2 dim;
    image.GetDimensions(dim.x, dim.y);
    
    if (any(tid >= dim))
        return;

    // Linear index calculation
    uint32_t idx = tid.y * dim.x + tid.x;

    uint32_t p_foreground = 0;
    uint32_t p_background = 0;

    uint8_t info_foreground = 0;
    uint8_t info_left_background = 0;
    uint8_t info_right_background = 0;
    
    // Keep values unsigned to avoid any backend-dependent sign extension.
    uint32_t buffer[4] = { 0u, 0u, 0u, 0u };

    buffer[0] = image[tid].x;
    buffer[1] = image[tid + uint2(1, 0)].x;
    buffer[2] = image[tid + uint2(0, 1)].x;
    buffer[3] = image[tid + uint2(1, 1)].x;

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
    if (has_bit(p_foreground, 0) && (image[tid - uint2(1, 1)].x == 255u))
    {
        father_offset_foreground = -(2 * int(dim.x) + 2);
    }

    // Q Square
    if ((has_bit(p_foreground, 1) && image[tid - uint2(0, 1)].x == 255u) ||
        (has_bit(p_foreground, 2) && image[tid + uint2(1, 0) - uint2(0, 1)].x == 255u))
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

    if ((has_bit(p_background, 1) && image[tid - uint2(0, 1)].x == 0u))
    {
        father_offset_left_background = -2 * int(dim.x);
    }
    if ((has_bit(p_background, 2) && image[tid + uint2(1, 0) - uint2(0, 1)].x == 0u))
    {
        father_offset_right_background = -2 * int(dim.x);
    }

    // R Square
    if (has_bit(p_foreground, 3) && image[tid + uint2(2, 0) - uint2(0, 1)].x == 255u)
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
    if ((has_bit(p_foreground, 4) && image[tid - uint2(1, 0)].x == 255u) ||
        (has_bit(p_foreground, 8) && image[tid + uint2(0, 1) - uint2(1, 0)].x == 255u))
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
    if ((has_bit(p_background, 4) && image[tid - uint2(1, 0)].x == 0u) ||
        (has_bit(p_background, 8) && image[tid + uint2(0, 1) - uint2(1, 0)].x == 0u))
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

void compression(uint2 tid, RWStructuredBuffer<uint32_t> labels, uint2 dim)
{
    tid *= 2;
    if (any(tid >= dim))
        return;

    uint32_t idx = tid.y * dim.x + tid.x;
    uint32_t background_labels_index = (tid.y + 1) * dim.x + tid.x;

    find_and_compress(labels, idx);
    find_and_compress(labels, background_labels_index);
    find_and_compress(labels, background_labels_index + 1);
}

void merge(uint2 tid, RWStructuredBuffer<uint32_t> labels, uint2 dim)
{
    tid *= 2;
    if (any(tid >= dim))
        return;

    uint32_t foreground_labels_index = tid.y * dim.x + tid.x;
    uint32_t background_labels_index = (tid.y + 1) * dim.x + tid.x;

    // FIX: Read info from the specific offset (index + 1)
    uint32_t info = labels[foreground_labels_index + 1];
    
    uint8_t info_foreground = uint8_t(info & 0xff);
    uint8_t info_left_background = uint8_t((info >> 8) & 0xff);
    uint8_t info_right_background = uint8_t((info >> 16) & 0xff);

    if (has_bit(info_foreground, Info::Q))
    {
        union_sets(labels, foreground_labels_index, foreground_labels_index - 2 * dim.x);
    }

    if (has_bit(info_foreground, Info::R))
    {
        union_sets(labels, foreground_labels_index, foreground_labels_index - 2 * dim.x + 2);
    }

    if (has_bit(info_foreground, Info::S))
    {
        union_sets(labels, foreground_labels_index, foreground_labels_index - 2);
    }

    if (has_bit(info_left_background, Info::S))
    {
        union_sets(labels, background_labels_index, background_labels_index - 1);
    }
    if (has_bit(info_right_background, Info::S))
    {
        union_sets(labels, background_labels_index + 1, background_labels_index);
    }
}

void final_labeling(uint2 tid, RWStructuredBuffer<uint32_t> labels,
                    RWStructuredBuffer<uint32_t> union_markers_size_device, uint2 dim)
{
    tid *= 2;
    if (any(tid >= dim))
        return;

    uint32_t foreground_labels_index = tid.y * dim.x + tid.x;
    uint32_t background_labels_index = (tid.y + 1) * dim.x + tid.x;

    // FIX: Separate reads for Label (low) and Info (high/neighbor)
    uint32_t foreground_label = labels[foreground_labels_index];
    uint32_t foreground_info_packed = labels[foreground_labels_index + 1]; // This contains the bitmasks

    uint32_t background_left_label = labels[background_labels_index];
    uint32_t background_right_label = labels[background_labels_index + 1];

    // FIX: Extract info bits from the packed 32-bit word read above
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

        // FIX: Explicit separate writes
        labels[foreground_labels_index] = a_label;
        labels[foreground_labels_index + 1] = b_label;
        
        labels[background_labels_index] = c_label;
        labels[background_labels_index + 1] = d_label;
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

        // FIX: Explicit separate writes
        labels[foreground_labels_index] = a_label;
        labels[foreground_labels_index + 1] = b_label;
        
        labels[background_labels_index] = c_label;
        labels[background_labels_index + 1] = d_label;
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
    {
        return;
    }

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


void init_buffer(uint2 tid, StructuredBuffer<uint8_t> image, ImageSize image_size, RWStructuredBuffer<uint32_t> labels)
{
    tid *= 2;
    uint2 dim = uint2(image_size.width, image_size.height);
    
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

    buffer[0] = image_u8_read(image, image_size, tid);
    buffer[1] = image_u8_read(image, image_size, tid + uint2(1, 0));
    buffer[2] = image_u8_read(image, image_size, tid + uint2(0, 1));
    buffer[3] = image_u8_read(image, image_size, tid + uint2(1, 1));

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
        is_white_sample_u32(image_u8_read(image, image_size, tid - uint2(1, 1))))
    {
        father_offset_foreground = -(2 * int(dim.x) + 2);
    }

    // Q Square
    bool q_foreground =
        (tid.y > 0u &&
         has_bit(p_foreground, 1) &&
         is_white_sample_u32(image_u8_read(image, image_size, tid - uint2(0, 1)))) ||
        (tid.y > 0u &&
         tid.x + 1u < dim.x &&
         has_bit(p_foreground, 2) &&
         is_white_sample_u32(image_u8_read(image, image_size, tid + uint2(1, 0) - uint2(0, 1))));
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
        image_u8_read(image, image_size, tid - uint2(0, 1)) == 0u)
    {
        father_offset_left_background = -2 * int(dim.x);
    }
    if (tid.y > 0u &&
        tid.x + 1u < dim.x &&
        has_bit(p_background, 2) &&
        image_u8_read(image, image_size, tid + uint2(1, 0) - uint2(0, 1)) == 0u)
    {
        father_offset_right_background = -2 * int(dim.x);
    }

    // R Square
    if (tid.y > 0u &&
        tid.x + 2u < dim.x &&
        has_bit(p_foreground, 3) &&
        is_white_sample_u32(image_u8_read(image, image_size, tid + uint2(2, 0) - uint2(0, 1))))
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
         is_white_sample_u32(image_u8_read(image, image_size, tid - uint2(1, 0)))) ||
        (tid.x > 0u &&
         tid.y + 1u < dim.y &&
         has_bit(p_foreground, 8) &&
         is_white_sample_u32(image_u8_read(image, image_size, tid + uint2(0, 1) - uint2(1, 0))));
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
         image_u8_read(image, image_size, tid - uint2(1, 0)) == 0u) ||
        (tid.x > 0u &&
         tid.y + 1u < dim.y &&
         has_bit(p_background, 8) &&
         image_u8_read(image, image_size, tid + uint2(0, 1) - uint2(1, 0)) == 0u);
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
