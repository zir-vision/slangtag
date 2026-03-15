StructuredBuffer<uint32_t> input_records;
RWStructuredBuffer<uint32_t> keys_out;
RWStructuredBuffer<uint32_t> indices_out;

struct PushConstants
{
    uint32_t valid_points;
    uint32_t words_per_record;
    uint32_t key_word_index;
    uint32_t key_transform;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

uint32_t key_to_sortable_u32(uint32_t key_word, uint32_t key_transform)
{
    if (key_transform == 1u)
    {
        return (key_word & 0x80000000u) != 0u ? ~key_word : (key_word ^ 0x80000000u);
    }
    return key_word;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint32_t tid = dispatch_thread_id.x;
    if (tid >= pc.valid_points)
        return;

    uint32_t record_base = tid * pc.words_per_record;
    uint32_t raw_key = input_records[record_base + pc.key_word_index];
    keys_out[tid] = key_to_sortable_u32(raw_key, pc.key_transform);
    indices_out[tid] = tid;
}
