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
