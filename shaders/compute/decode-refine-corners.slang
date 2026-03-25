import "./image_buffer.slang";
import "./detect-control.slang";

StructuredBuffer<uint8_t> image_gray;
RWStructuredBuffer<uint32_t> decoded_tags_out;
StructuredBuffer<uint32_t> control;

static const uint32_t kDecodedTagWordsPerTag = 19u;
static const uint32_t kDecodedTagCornersStartWord = 4u;

struct PushConstants
{
    ImageSize image_size;
    uint32_t max_iterations;
    uint32_t samples_per_edge;
    float search_radius_px;
    float min_gradient;
    float min_det;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

struct RefinedLine
{
    float2 point;
    float2 normal;
};

[ForceInline]
uint32_t decoded_word_index(uint32_t tag_index, uint32_t word_index)
{
    return tag_index * kDecodedTagWordsPerTag + word_index;
}

[ForceInline]
float2 load_corner(uint32_t tag_index, uint32_t corner_index)
{
    uint32_t base = kDecodedTagCornersStartWord + corner_index * 2u;
    float x = asfloat(decoded_tags_out[decoded_word_index(tag_index, base + 0u)]);
    float y = asfloat(decoded_tags_out[decoded_word_index(tag_index, base + 1u)]);
    return float2(x, y);
}

[ForceInline]
void store_corner(uint32_t tag_index, uint32_t corner_index, float2 corner)
{
    uint32_t base = kDecodedTagCornersStartWord + corner_index * 2u;
    decoded_tags_out[decoded_word_index(tag_index, base + 0u)] = asuint(corner.x);
    decoded_tags_out[decoded_word_index(tag_index, base + 1u)] = asuint(corner.y);
}

[ForceInline]
bool finite2(float2 v)
{
    return (v.x == v.x) && (v.y == v.y) && abs(v.x) < 1.0e20f && abs(v.y) < 1.0e20f;
}

[ForceInline]
float sample_luma(float2 p)
{
    int x = int(floor(p.x + 0.5f));
    int y = int(floor(p.y + 0.5f));
    return float(image_u8_read_i32(image_gray, pc.image_size, x, y));
}

[ForceInline]
float2 sample_grad(float2 p)
{
    float gx = 0.5f * (sample_luma(p + float2(1.0f, 0.0f)) - sample_luma(p - float2(1.0f, 0.0f)));
    float gy = 0.5f * (sample_luma(p + float2(0.0f, 1.0f)) - sample_luma(p - float2(0.0f, 1.0f)));
    return float2(gx, gy);
}

bool build_line(float2 c0, float2 c1, out RefinedLine line_out)
{
    float2 edge = c1 - c0;
    float edge_len = length(edge);
    if (edge_len < 1e-6f)
        return false;

    float2 normal = float2(edge.y, -edge.x) / edge_len;
    uint32_t samples = max(pc.samples_per_edge, 2u);
    float sum_w = 0.0f;
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float sum_xx = 0.0f;
    float sum_xy = 0.0f;
    float sum_yy = 0.0f;
    uint32_t used = 0u;

    uint32_t search_steps = max(3u, uint32_t(ceil(pc.search_radius_px * 2.0f)) + 1u);
    float radius = pc.search_radius_px;
    float denom = float(max(search_steps - 1u, 1u));

    for (uint32_t s = 0u; s < samples; ++s)
    {
        float t = (float(s) + 0.5f) / float(samples);
        float2 base = lerp(c0, c1, t);
        float best_score = -1.0f;
        float2 best_point = base;

        for (uint32_t si = 0u; si < search_steps; ++si)
        {
            float offset = -radius + (2.0f * radius * float(si)) / denom;
            float2 candidate = base + normal * offset;
            float2 grad = sample_grad(candidate);
            float score = abs(dot(grad, normal));
            if (score > best_score)
            {
                best_score = score;
                best_point = candidate;
            }
        }

        if (best_score < pc.min_gradient)
            continue;

        float w = best_score;
        sum_w += w;
        sum_x += w * best_point.x;
        sum_y += w * best_point.y;
        sum_xx += w * best_point.x * best_point.x;
        sum_xy += w * best_point.x * best_point.y;
        sum_yy += w * best_point.y * best_point.y;
        used += 1u;
    }

    if (used < 2u || sum_w <= 0.0f)
        return false;

    float mx = sum_x / sum_w;
    float my = sum_y / sum_w;
    float cxx = max(0.0f, sum_xx / sum_w - mx * mx);
    float cxy = sum_xy / sum_w - mx * my;
    float cyy = max(0.0f, sum_yy / sum_w - my * my);

    float tr = cxx + cyy;
    float det = cxx * cyy - cxy * cxy;
    float disc = max(0.0f, tr * tr * 0.25f - det);
    float lambda = tr * 0.5f + sqrt(disc);

    float2 tangent = float2(cxy, lambda - cxx);
    if (dot(tangent, tangent) < 1e-10f)
    {
        tangent = float2(lambda - cyy, cxy);
    }
    float tlen = length(tangent);
    if (tlen < 1e-6f)
        return false;
    tangent /= tlen;

    line_out.point = float2(mx, my);
    line_out.normal = float2(-tangent.y, tangent.x);
    return finite2(line_out.point) && finite2(line_out.normal);
}

bool intersect_lines(RefinedLine a, RefinedLine b, out float2 p_out)
{
    float d0 = dot(a.normal, a.point);
    float d1 = dot(b.normal, b.point);
    float det = a.normal.x * b.normal.y - a.normal.y * b.normal.x;
    if (abs(det) < pc.min_det)
        return false;

    p_out.x = (d0 * b.normal.y - a.normal.y * d1) / det;
    p_out.y = (a.normal.x * d1 - d0 * b.normal.x) / det;
    return finite2(p_out);
}

bool validate_quad(float2 corners[4])
{
    [ForceUnroll]
    for (uint32_t i = 0u; i < 4u; ++i)
    {
        if (!finite2(corners[i]))
            return false;
    }

    float area2 = 0.0f;
    for (uint32_t i = 0u; i < 4u; ++i)
    {
        uint32_t j = (i + 1u) & 3u;
        area2 += corners[i].x * corners[j].y - corners[j].x * corners[i].y;
    }
    if (abs(area2) < 1e-2f)
        return false;

    float sign_ref = 0.0f;
    for (uint32_t i = 0u; i < 4u; ++i)
    {
        uint32_t i0 = i;
        uint32_t i1 = (i + 1u) & 3u;
        uint32_t i2 = (i + 2u) & 3u;
        float2 e0 = corners[i1] - corners[i0];
        float2 e1 = corners[i2] - corners[i1];
        float cross_z = e0.x * e1.y - e0.y * e1.x;
        if (abs(cross_z) < 1e-4f)
            return false;
        if (i == 0u)
        {
            sign_ref = cross_z;
        }
        else if (sign_ref * cross_z <= 0.0f)
        {
            return false;
        }
    }
    return true;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint32_t tag_index = dispatch_thread_id.x;
    uint32_t decoded_count = control[kControlDecodedTagCountWord];
    if (tag_index >= decoded_count)
        return;

    float2 original[4];
    [ForceUnroll]
    for (uint32_t i = 0u; i < 4u; ++i)
    {
        original[i] = load_corner(tag_index, i);
    }

    float2 current[4];
    [ForceUnroll]
    for (uint32_t i = 0u; i < 4u; ++i)
    {
        current[i] = original[i];
    }

    bool ok = true;
    uint32_t iterations = max(pc.max_iterations, 1u);
    for (uint32_t iter = 0u; iter < iterations; ++iter)
    {
        RefinedLine lines[4];
        for (uint32_t e = 0u; e < 4u; ++e)
        {
            uint32_t n = (e + 1u) & 3u;
            if (!build_line(current[e], current[n], lines[e]))
            {
                ok = false;
                break;
            }
        }
        if (!ok)
            break;

        float2 next_corners[4];
        for (uint32_t i = 0u; i < 4u; ++i)
        {
            uint32_t n = (i + 1u) & 3u;
            if (!intersect_lines(lines[i], lines[n], next_corners[n]))
            {
                ok = false;
                break;
            }
        }
        if (!ok || !validate_quad(next_corners))
        {
            ok = false;
            break;
        }

        [ForceUnroll]
        for (uint32_t i = 0u; i < 4u; ++i)
        {
            current[i] = next_corners[i];
        }
    }

    if (!ok)
    {
        [ForceUnroll]
        for (uint32_t i = 0u; i < 4u; ++i)
        {
            current[i] = original[i];
        }
    }

    [ForceUnroll]
    for (uint32_t i = 0u; i < 4u; ++i)
    {
        store_corner(tag_index, i, current[i]);
    }
}
