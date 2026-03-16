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
