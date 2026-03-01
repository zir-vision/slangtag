import "../sort.slang";

StructuredBuffer<uint32_t> input;
RWStructuredBuffer<uint32_t> output;
uint32_t valid_points;
uint32_t total_points;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    prepare_selected_blob_points(dispatch_thread_id.x, input, output, valid_points, total_points);
}
