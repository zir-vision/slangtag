import "../select.slang";

StructuredBuffer<uint32_t> input;
RWStructuredBuffer<uint32_t> output;
RWStructuredBuffer<uint32_t> output_count;

struct PushConstants
{
    uint32_t total_points;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint tid = dispatch_thread_id.x;
    if (tid >= pc.total_points)
        return;
    if (!blob_diff_point_is_nonzero(input, tid))
        return;

    uint32_t output_index;
    InterlockedAdd(output_count[0], 1u, output_index);
    uint32_t input_word_index = tid * kBlobDiffWordsPerPoint;
    uint32_t output_word_index = output_index * kBlobDiffWordsPerPoint;
    [ForceUnroll]
    for (uint32_t i = 0u; i < kBlobDiffWordsPerPoint; ++i)
    {
        output[output_word_index + i] = input[input_word_index + i];
    }
}
