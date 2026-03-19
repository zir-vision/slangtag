static const uint32_t kControlBlobDiffFilteredCountWord = 0u;
static const uint32_t kControlBlobExtentCountWord = 1u;
static const uint32_t kControlSelectedBlobExtentCountWord = 2u;
static const uint32_t kControlSelectedBlobPointCountWord = 3u;
static const uint32_t kControlPeakExtentCountWord = 4u;
static const uint32_t kControlFittedQuadCountWord = 5u;

static const uint32_t kControlDispatchBuildBlobPairExtentsWord = 16u;
static const uint32_t kControlDispatchRadixBlobInitWord = 19u;
static const uint32_t kControlDispatchRadixBlobUpdateWord = 22u;
static const uint32_t kControlDispatchRadixBlobGatherWord = 25u;
static const uint32_t kControlDispatchRewriteSelectedBlobPointsWord = 28u;
static const uint32_t kControlDispatchRadixSelectedInitWord = 31u;
static const uint32_t kControlDispatchRadixSelectedUpdateWord = 34u;
static const uint32_t kControlDispatchRadixSelectedGatherWord = 37u;
static const uint32_t kControlDispatchRebuildFilteredExtentStartsWord = 40u;
static const uint32_t kControlDispatchBuildLineFitPointsWord = 43u;
static const uint32_t kControlDispatchFitLinePass0Word = 46u;
static const uint32_t kControlDispatchFitLinePass1Word = 49u;
static const uint32_t kControlDispatchFitLinePass2Word = 52u;
static const uint32_t kControlDispatchRadixPeaksInitWord = 55u;
static const uint32_t kControlDispatchRadixPeaksUpdateWord = 58u;
static const uint32_t kControlDispatchRadixPeaksGatherWord = 61u;
static const uint32_t kControlDispatchBuildPeakExtentsWord = 64u;
static const uint32_t kControlDispatchFitQuadsWord = 67u;
static const uint32_t kControlDispatchPrepareDecodeQuadsWord = 70u;
static const uint32_t kControlDispatchExtractCandidateBitsWord = 73u;

[ForceInline]
uint32_t detect_dispatch_groups_1d(uint32_t total_invocations, uint32_t local_size_x)
{
    if (total_invocations == 0u)
        return 1u;
    return (total_invocations + local_size_x - 1u) / local_size_x;
}

[ForceInline]
void write_dispatch_1d(
    RWStructuredBuffer<uint32_t> control,
    uint32_t base_word,
    uint32_t total_invocations,
    uint32_t local_size_x)
{
    control[base_word + 0u] = detect_dispatch_groups_1d(total_invocations, local_size_x);
    control[base_word + 1u] = 1u;
    control[base_word + 2u] = 1u;
}
