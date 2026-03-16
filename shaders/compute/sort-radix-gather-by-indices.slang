StructuredBuffer<uint32_t> input_records;
StructuredBuffer<uint32_t> indices_in;
RWStructuredBuffer<uint32_t> output_records;

struct PushConstants
{
    uint32_t valid_points;
    uint32_t words_per_record;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint32_t tid = dispatch_thread_id.x;
    if (tid >= pc.valid_points)
        return;

    uint32_t src_record_index = indices_in[tid];
    uint32_t src_base = src_record_index * pc.words_per_record;
    uint32_t dst_base = tid * pc.words_per_record;

    for (uint32_t i = 0u; i < pc.words_per_record; ++i)
    {
        output_records[dst_base + i] = input_records[src_base + i];
    }
}
