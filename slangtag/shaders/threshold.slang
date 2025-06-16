

void decimate(uint2 tid, Texture2D<uint1> input_texture,
              RWTexture2D<uint1> decimated_texture)
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

uint2 minmax_row(uint4 row)
{
    uint min_val = min(min(row.x, row.y), min(row.z, row.w));
    uint max_val = max(max(row.x, row.y), max(row.z, row.w));
    return uint2(min_val, max_val);
}

uint2 minmax_minmax(uint2 a, uint2 b)
{
    return uint2(min(a.x, b.x), max(a.y, b.y));
}

// [numthreads(16, 16, 1)]
void minmax(uint2 tid, Texture2D<uint1> input_texture,
            RWTexture2D<uint2> minmax_texture)
{
    uint2 dim;
    input_texture.GetDimensions(dim.x, dim.y);
    uint2 pixel = tid.xy;
    if (any(pixel >= dim))
        return;

    uint2 vals[4];

    for (int i = 0; i < 4; ++i)
    {
        uint4 row = input_texture[(pixel * 4) + uint2(0, i)];
        vals[i] = minmax_row(row);
    }

    minmax_texture[pixel] = minmax_minmax(
        minmax_minmax(vals[0], vals[1]),
        minmax_minmax(vals[2], vals[3]));
}

void filter_minmax(uint2 tid, Texture2D<uint2> minmax_texture,
                   RWTexture2D<uint2> filtered_texture)
{
    uint2 dim;
    minmax_texture.GetDimensions(dim.x, dim.y);
    uint2 pixel = tid.xy;
    if (any(pixel >= dim))
        return;

    uint2 result = uint2(255, 0);

    [ForceUnroll]
    for (int i = -1; i <= 1; ++i)
    {
        [ForceUnroll]
        for (int j = -1; j <= 1; ++j)
        {
            uint2 neighbor_pixel = pixel + uint2(i, j);
            if (any(neighbor_pixel >= dim))
                continue;

            uint2 neighbor_minmax = minmax_texture[neighbor_pixel];

            result = minmax_minmax(result, neighbor_minmax);
        }
    }

    filtered_texture[pixel] = result;
}

void threshold(uint2 tid,
               Texture2D<uint1> decimated_texture,
               Texture2D<uint2> filtered_texture,
               RWTexture2D<uint1> thresholded_texture, uint min_white_black_diff)
{
    uint2 dim;
    decimated_texture.GetDimensions(dim.x, dim.y);
    uint2 pixel = tid.xy;
    if (any(pixel >= dim))
        return;

    uint2 minmax = filtered_texture[pixel / 4];

    uint result = 0;

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
