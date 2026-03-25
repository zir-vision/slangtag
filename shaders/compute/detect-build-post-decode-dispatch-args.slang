import "./detect-control.slang";

StructuredBuffer<uint32_t> control_in;
RWStructuredBuffer<uint32_t> control_out;

struct PushConstants
{
    uint32_t one_d_local_size_x;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    if (dispatch_thread_id.x != 0u || dispatch_thread_id.y != 0u || dispatch_thread_id.z != 0u)
        return;

    uint32_t one_d_local_size_x = max(pc.one_d_local_size_x, 1u);
    uint32_t decoded_tag_count = control_in[kControlDecodedTagCountWord];
    write_dispatch_1d(
        control_out,
        kControlDispatchDecodeRefineCornersWord,
        decoded_tag_count,
        one_d_local_size_x);
}
