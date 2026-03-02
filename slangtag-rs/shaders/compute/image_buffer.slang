struct ImageSize
{
    uint32_t width;
    uint32_t height;
};

[ForceInline]
bool image_in_bounds(ImageSize size, uint2 p)
{
    return p.x < size.width && p.y < size.height;
}

[ForceInline]
uint32_t image_linear_index(ImageSize size, uint2 p)
{
    return p.y * size.width + p.x;
}

[ForceInline]
uint8_t image_u8_read(StructuredBuffer<uint8_t> image, ImageSize size, uint2 p)
{
    if (!image_in_bounds(size, p))
        return 0u;
    return image[image_linear_index(size, p)];
}

[ForceInline]
uint8_t image_u8_read_i32(StructuredBuffer<uint8_t> image, ImageSize size, int x, int y)
{
    if (x < 0 || y < 0 || x >= int(size.width) || y >= int(size.height))
        return 0u;
    return image_u8_read(image, size, uint2(uint32_t(x), uint32_t(y)));
}

[ForceInline]
void image_u8_write(RWStructuredBuffer<uint8_t> image, ImageSize size, uint2 p, uint8_t value)
{
    if (!image_in_bounds(size, p))
        return;
    image[image_linear_index(size, p)] = value;
}

[ForceInline]
uint8_t2 image_rg8_read(StructuredBuffer<uint8_t2> image, ImageSize size, uint2 p)
{
    if (!image_in_bounds(size, p))
        return uint8_t2(0u, 0u);
    return image[image_linear_index(size, p)];
}

[ForceInline]
void image_rg8_write(RWStructuredBuffer<uint8_t2> image, ImageSize size, uint2 p, uint8_t2 value)
{
    if (!image_in_bounds(size, p))
        return;
    image[image_linear_index(size, p)] = value;
}
