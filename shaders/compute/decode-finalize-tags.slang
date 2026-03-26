import "./detect-control.slang";

StructuredBuffer<uint32_t> bits;
StructuredBuffer<uint32_t> fitted_quads;
StructuredBuffer<uint32_t> apriltag_lookup;
RWStructuredBuffer<uint32_t> decoded_tags_out;
RWStructuredBuffer<uint32_t> control;

static const uint32_t kFittedQuadWordsPerQuad = 15u;
static const uint32_t kFittedQuadBlobIndexWord = 0u;
static const uint32_t kFittedQuadReversedBorderWord = 1u;
static const uint32_t kFittedQuadScoreWord = 2u;
static const uint32_t kFittedQuadCorner0XWord = 3u;

static const uint32_t kDecodedTagWordsPerTag = 19u;
static const uint32_t kDecodedTagQuadIndexWord = 0u;
static const uint32_t kDecodedTagBlobIndexWord = 1u;
static const uint32_t kDecodedTagReversedBorderWord = 2u;
static const uint32_t kDecodedTagScoreWord = 3u;
static const uint32_t kDecodedTagCornersStartWord = 4u;
static const uint32_t kDecodedTagIdWord = 12u;
static const uint32_t kDecodedTagRotationWord = 13u;
static const uint32_t kDecodedTagPayloadLoWord = 14u;
static const uint32_t kDecodedTagPayloadHiWord = 15u;
static const uint32_t kDecodedTagBitsLoWord = 16u;
static const uint32_t kDecodedTagBitsHiWord = 17u;
static const uint32_t kDecodedTagFlagsWord = 18u;

struct PushConstants
{
    uint32_t marker_size_with_borders;
    uint32_t marker_border_bits;
    uint32_t apriltag_marker_size;
    uint32_t detect_inverted_marker;
    uint32_t max_border_errors;
    uint32_t max_corrected_bits;
    uint32_t lookup_entry_count;
    uint32_t decoded_tag_capacity;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[ForceInline]
uint32_t read_fitted_quad_word(StructuredBuffer<uint32_t> quads, uint32_t quad_index, uint32_t word_index)
{
    return quads[quad_index * kFittedQuadWordsPerQuad + word_index];
}

[ForceInline]
void write_decoded_word(RWStructuredBuffer<uint32_t> dst, uint32_t tag_index, uint32_t word_index, uint32_t value)
{
    dst[tag_index * kDecodedTagWordsPerTag + word_index] = value;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint32_t quad_index = dispatch_thread_id.x;
    uint32_t quad_count = control[kControlFittedQuadCountWord];
    if (quad_index >= quad_count)
        return;

    uint32_t marker_size = pc.marker_size_with_borders;
    uint32_t border = pc.marker_border_bits;
    uint32_t inner_size = pc.apriltag_marker_size;
    uint32_t cells_per_quad = marker_size * marker_size;
    if (cells_per_quad == 0u || inner_size == 0u || marker_size < (2u * border + 1u))
        return;

    uint32_t bits_base = quad_index * cells_per_quad;
    uint32_t border_errors = 0u;
    uint32_t inverted_border_errors = 0u;
    uint32_t payload_lo = 0u;
    uint32_t payload_hi = 0u;
    uint32_t payload_inv_lo = 0u;
    uint32_t payload_inv_hi = 0u;
    uint32_t all_bits_lo = 0u;
    uint32_t all_bits_hi = 0u;
    uint32_t all_bits_inv_lo = 0u;
    uint32_t all_bits_inv_hi = 0u;
    uint32_t payload_bit_index = 0u;

    for (uint32_t y = 0u; y < marker_size; ++y)
    {
        for (uint32_t x = 0u; x < marker_size; ++x)
        {
            uint32_t cell_index = y * marker_size + x;
            uint32_t bit = bits[bits_base + cell_index] != 0u ? 1u : 0u;
            uint32_t inv = bit ^ 1u;

            if (cell_index < 32u)
            {
                all_bits_lo |= (bit << cell_index);
                all_bits_inv_lo |= (inv << cell_index);
            }
            else
            {
                uint32_t shift = cell_index - 32u;
                all_bits_hi |= (bit << shift);
                all_bits_inv_hi |= (inv << shift);
            }

            bool is_border = x < border || x >= (marker_size - border) || y < border || y >= (marker_size - border);
            if (is_border)
            {
                border_errors += bit;
                inverted_border_errors += inv;
                continue;
            }

            if (payload_bit_index < 32u)
            {
                payload_lo |= (bit << payload_bit_index);
                payload_inv_lo |= (inv << payload_bit_index);
            }
            else
            {
                uint32_t shift = payload_bit_index - 32u;
                payload_hi |= (bit << shift);
                payload_inv_hi |= (inv << shift);
            }
            payload_bit_index += 1u;
        }
    }

    bool use_inverted = pc.detect_inverted_marker != 0u && inverted_border_errors < border_errors;
    uint32_t selected_border_errors = use_inverted ? inverted_border_errors : border_errors;
    if (selected_border_errors > pc.max_border_errors)
        return;

    uint32_t selected_payload_lo = use_inverted ? payload_inv_lo : payload_lo;
    uint32_t selected_payload_hi = use_inverted ? payload_inv_hi : payload_hi;
    uint32_t selected_bits_lo = use_inverted ? all_bits_inv_lo : all_bits_lo;
    uint32_t selected_bits_hi = use_inverted ? all_bits_inv_hi : all_bits_hi;

    uint32_t best_distance = 0xFFFFFFFFu;
    uint32_t best_id = 0u;
    uint32_t best_rotation = 0u;
    for (uint32_t i = 0u; i < pc.lookup_entry_count; ++i)
    {
        uint32_t base = i * 4u;
        uint32_t code_lo = apriltag_lookup[base + 0u];
        uint32_t code_hi = apriltag_lookup[base + 1u];
        uint32_t id = apriltag_lookup[base + 2u];
        uint32_t rotation = apriltag_lookup[base + 3u];
        uint32_t distance = countbits(selected_payload_lo ^ code_lo) + countbits(selected_payload_hi ^ code_hi);
        if (distance < best_distance)
        {
            best_distance = distance;
            best_id = id;
            best_rotation = rotation;
            if (distance == 0u)
                break;
        }
    }
    if (best_distance > pc.max_corrected_bits)
        return;

    uint32_t decoded_index = 0u;
    InterlockedAdd(control[kControlDecodedTagCountWord], 1u, decoded_index);
    if (decoded_index >= pc.decoded_tag_capacity)
        return;

    write_decoded_word(decoded_tags_out, decoded_index, kDecodedTagQuadIndexWord, quad_index);
    write_decoded_word(decoded_tags_out, decoded_index, kDecodedTagBlobIndexWord, read_fitted_quad_word(fitted_quads, quad_index, kFittedQuadBlobIndexWord));
    write_decoded_word(decoded_tags_out, decoded_index, kDecodedTagReversedBorderWord, read_fitted_quad_word(fitted_quads, quad_index, kFittedQuadReversedBorderWord));
    write_decoded_word(decoded_tags_out, decoded_index, kDecodedTagScoreWord, read_fitted_quad_word(fitted_quads, quad_index, kFittedQuadScoreWord));
    [ForceUnroll]
    for (uint32_t corner_word = 0u; corner_word < 8u; ++corner_word)
    {
        write_decoded_word(
            decoded_tags_out,
            decoded_index,
            kDecodedTagCornersStartWord + corner_word,
            read_fitted_quad_word(fitted_quads, quad_index, kFittedQuadCorner0XWord + corner_word));
    }
    write_decoded_word(decoded_tags_out, decoded_index, kDecodedTagIdWord, best_id);
    write_decoded_word(decoded_tags_out, decoded_index, kDecodedTagRotationWord, best_rotation);
    write_decoded_word(decoded_tags_out, decoded_index, kDecodedTagPayloadLoWord, selected_payload_lo);
    write_decoded_word(decoded_tags_out, decoded_index, kDecodedTagPayloadHiWord, selected_payload_hi);
    write_decoded_word(decoded_tags_out, decoded_index, kDecodedTagBitsLoWord, selected_bits_lo);
    write_decoded_word(decoded_tags_out, decoded_index, kDecodedTagBitsHiWord, selected_bits_hi);
    write_decoded_word(decoded_tags_out, decoded_index, kDecodedTagFlagsWord, use_inverted ? 1u : 0u);
}
