import "compute/image_buffer.slang";

void decimate(uint2 tid, Texture2D<uint8_t1> input_texture,
              RWTexture2D<uint8_t1> decimated_texture)
{
    uint2 dim;
    input_texture.GetDimensions(dim.x, dim.y);
    uint2 pixel = tid.xy;
    if (any(pixel >= dim))
        return;

    if (pixel.x % 2 == 0 && pixel.y % 2 == 0)
    {
        uint2 out_pixel = pixel / 2;
        decimated_texture[out_pixel] = input_texture[pixel];
    }
}

uint8_t2 minmax_row(uint8_t4 row)
{
    uint8_t min_val = min(min(row.x, row.y), min(row.z, row.w));
    uint8_t max_val = max(max(row.x, row.y), max(row.z, row.w));
    return uint8_t2(min_val, max_val);
}

uint8_t2 minmax_minmax(uint8_t2 a, uint8_t2 b)
{
    return uint8_t2(min(a.x, b.x), max(a.y, b.y));
}

// [numthreads(16, 16, 1)]
void minmax(uint2 tid, Texture2D<uint8_t1> input_texture,
            RWTexture2D<uint8_t2> minmax_texture)
{
    uint2 dim;
    minmax_texture.GetDimensions(dim.x, dim.y);
    uint2 pixel = tid.xy;
    if (any(pixel >= dim))
        return;

    uint8_t2 vals[4];

    for (int i = 0; i < 4; ++i)
    {
        uint2 row_base = (pixel * 4) + uint2(0, i);
        uint8_t4 row = uint8_t4(
            input_texture[row_base + uint2(0, 0)].x,
            input_texture[row_base + uint2(1, 0)].x,
            input_texture[row_base + uint2(2, 0)].x,
            input_texture[row_base + uint2(3, 0)].x);
        vals[i] = minmax_row(row);
    }

    minmax_texture[pixel] = minmax_minmax(
        minmax_minmax(vals[0], vals[1]),
        minmax_minmax(vals[2], vals[3]));
}

void filter_minmax(uint2 tid, Texture2D<uint8_t2> minmax_texture,
                   RWTexture2D<uint8_t2> filtered_texture)
{
    uint2 dim;
    minmax_texture.GetDimensions(dim.x, dim.y);
    uint2 pixel = tid.xy;
    if (any(pixel >= dim))
        return;

    uint8_t2 result = uint8_t2(255, 0);

    [ForceUnroll]
    for (int i = -1; i <= 1; ++i)
    {
        [ForceUnroll]
        for (int j = -1; j <= 1; ++j)
        {
            uint2 neighbor_pixel = pixel + uint2(i, j);
            if (any(neighbor_pixel >= dim))
                continue;

            uint8_t2 neighbor_minmax = minmax_texture[neighbor_pixel];

            result = minmax_minmax(result, neighbor_minmax);
        }
    }

    filtered_texture[pixel] = result;
}

void threshold(uint2 tid,
               Texture2D<uint8_t1> decimated_texture,
               Texture2D<uint8_t2> filtered_texture,
               RWTexture2D<uint8_t1> thresholded_texture, uint min_white_black_diff)
{
    uint2 dim;
    decimated_texture.GetDimensions(dim.x, dim.y);
    uint2 pixel = tid.xy;
    if (any(pixel >= dim))
        return;

    uint2 minmax = filtered_texture[pixel / 4];

    uint8_t result = 0;

    if (minmax.y - minmax.x < min_white_black_diff)
    {
        result = 127;
    }
    else
    {
        uint threshold_value = minmax.x + (minmax.y - minmax.x) / 2;
        if (decimated_texture[pixel] > threshold_value)
        {
            result = 255;
        }
        else
        {
            result = 0;
        }
    }

    thresholded_texture[pixel] = result;
}

void decimate_buffer(uint2 tid,
                     StructuredBuffer<uint8_t> input_image,
                     ImageSize input_size,
                     RWStructuredBuffer<uint8_t> decimated_image,
                     ImageSize decimated_size)
{
    uint2 pixel = tid.xy;
    if (!image_in_bounds(input_size, pixel))
        return;

    let x_decimate = input_size.width / decimated_size.width;
    let y_decimate = input_size.height / decimated_size.height;
    
    if (x_decimate != y_decimate)
    {
        // For simplicity, we require the decimation factor to be the same in both dimensions.
        return;
    }

    let decimate_factor = x_decimate;

    if (pixel.x % decimate_factor == 0u && pixel.y % decimate_factor == 0u)
    {
        uint2 out_pixel = pixel / decimate_factor;
        if (image_in_bounds(decimated_size, out_pixel))
        {
            image_u8_write(
                decimated_image,
                decimated_size,
                out_pixel,
                image_u8_read(input_image, input_size, pixel));
        }
    }
}

void minmax_buffer(uint2 tid,
                   StructuredBuffer<uint8_t> input_image,
                   ImageSize input_size,
                   RWStructuredBuffer<uint8_t2> minmax_image,
                   ImageSize minmax_size)
{
    uint2 pixel = tid.xy;
    if (!image_in_bounds(minmax_size, pixel))
        return;

    uint8_t2 vals[4];
    for (int i = 0; i < 4; ++i)
    {
        uint2 row_base = (pixel * 4u) + uint2(0u, uint32_t(i));
        uint8_t4 row = uint8_t4(
            image_u8_read(input_image, input_size, row_base + uint2(0u, 0u)),
            image_u8_read(input_image, input_size, row_base + uint2(1u, 0u)),
            image_u8_read(input_image, input_size, row_base + uint2(2u, 0u)),
            image_u8_read(input_image, input_size, row_base + uint2(3u, 0u)));
        vals[i] = minmax_row(row);
    }

    image_rg8_write(
        minmax_image,
        minmax_size,
        pixel,
        minmax_minmax(minmax_minmax(vals[0], vals[1]), minmax_minmax(vals[2], vals[3])));
}

void filter_minmax_buffer(uint2 tid,
                          StructuredBuffer<uint8_t2> minmax_image,
                          ImageSize minmax_size,
                          RWStructuredBuffer<uint8_t2> filtered_image,
                          ImageSize filtered_size)
{
    uint2 pixel = tid.xy;
    if (!image_in_bounds(minmax_size, pixel))
        return;

    uint8_t2 result = uint8_t2(255u, 0u);
    [ForceUnroll]
    for (int i = -1; i <= 1; ++i)
    {
        [ForceUnroll]
        for (int j = -1; j <= 1; ++j)
        {
            int nx = int(pixel.x) + i;
            int ny = int(pixel.y) + j;
            if (nx < 0 || ny < 0 || nx >= int(minmax_size.width) || ny >= int(minmax_size.height))
                continue;

            uint2 neighbor = uint2(uint32_t(nx), uint32_t(ny));
            result = minmax_minmax(result, image_rg8_read(minmax_image, minmax_size, neighbor));
        }
    }

    image_rg8_write(filtered_image, filtered_size, pixel, result);
}

void threshold_buffer(uint2 tid,
                      StructuredBuffer<uint8_t> decimated_image,
                      ImageSize decimated_size,
                      StructuredBuffer<uint8_t2> filtered_image,
                      ImageSize filtered_size,
                      RWStructuredBuffer<uint8_t> thresholded_image,
                      ImageSize thresholded_size,
                      uint min_white_black_diff)
{
    uint2 pixel = tid.xy;
    if (!image_in_bounds(decimated_size, pixel))
        return;

    uint2 minmax_pixel = pixel / 4u;
    if (!image_in_bounds(filtered_size, minmax_pixel))
        return;

    uint8_t2 minmax = image_rg8_read(filtered_image, filtered_size, minmax_pixel);
    uint8_t result = 0u;

    if (minmax.y - minmax.x < min_white_black_diff)
    {
        result = 127u;
    }
    else
    {
        uint threshold_value = uint(minmax.x) + (uint(minmax.y) - uint(minmax.x)) / 2u;
        result = image_u8_read(decimated_image, decimated_size, pixel) > threshold_value ? 255u : 0u;
    }

    image_u8_write(thresholded_image, thresholded_size, pixel, result);
}
