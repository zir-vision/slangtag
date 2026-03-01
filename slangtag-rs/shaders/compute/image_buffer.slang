struct ImageSize
{
    uint32_t width;
    uint32_t height;
};

bool image_in_bounds(ImageSize size, uint2 p)
{
    return p.x < size.width && p.y < size.height;
}

uint32_t image_linear_index(ImageSize size, uint2 p)
{
    return p.y * size.width + p.x;
}

uint8_t image_u8_read(StructuredBuffer<uint32_t> image, ImageSize size, uint2 p)
{
    if (!image_in_bounds(size, p))
        return 0u;
    return uint8_t(image[image_linear_index(size, p)] & 0xFFu);
}

uint8_t image_u8_read_i32(StructuredBuffer<uint32_t> image, ImageSize size, int x, int y)
{
    if (x < 0 || y < 0 || x >= int(size.width) || y >= int(size.height))
        return 0u;
    return image_u8_read(image, size, uint2(uint32_t(x), uint32_t(y)));
}

void image_u8_write(RWStructuredBuffer<uint32_t> image, ImageSize size, uint2 p, uint8_t value)
{
    if (!image_in_bounds(size, p))
        return;
    image[image_linear_index(size, p)] = uint32_t(value);
}

uint8_t2 image_rg8_read(StructuredBuffer<uint32_t> image, ImageSize size, uint2 p)
{
    if (!image_in_bounds(size, p))
        return uint8_t2(0u, 0u);

    uint32_t packed = image[image_linear_index(size, p)];
    return uint8_t2(
        uint8_t(packed & 0xFFu),
        uint8_t((packed >> 8u) & 0xFFu));
}

void image_rg8_write(RWStructuredBuffer<uint32_t> image, ImageSize size, uint2 p, uint8_t2 value)
{
    if (!image_in_bounds(size, p))
        return;

    uint32_t packed = uint32_t(value.x) | (uint32_t(value.y) << 8u);
    image[image_linear_index(size, p)] = packed;
}
