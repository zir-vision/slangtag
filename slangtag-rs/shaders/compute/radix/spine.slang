import "./constants.slang";

StructuredBuffer<uint> elementCounts : register(t0, space0);
RWStructuredBuffer<uint> globalHistogram : register(u1, space0);
RWStructuredBuffer<uint> partitionHistogram : register(u2, space0);

groupshared uint reduction;
groupshared uint intermediate[MAX_SUBGROUP_SIZE];

// dispatch this shader (RADIX, 1, 1), so that gl_WorkGroupID.x is radix
[shader("compute")]
[numthreads(WORKGROUP_SIZE)]
void main(uint3 groupThreadID: SV_GroupThreadID, uint3 groupId: SV_GroupID,
          uint groupIndex: SV_GroupIndex, uniform int pass) {
  printf("spine groupThreadID: (%u, %u, %u), groupId: (%u, %u, %u), groupIndex: %u\n", groupThreadID.x,
         groupThreadID.y, groupThreadID.z, groupId.x, groupId.y, groupId.z, groupIndex);
  uint elementCount = elementCounts[0];

  uint laneIndex = WaveGetLaneIndex();  // 0..31
  uint laneCount = WaveGetLaneCount();  // 32
  uint waveIndex = groupIndex / laneCount;
  uint waveCount = WORKGROUP_SIZE / laneCount;
  uint index = waveIndex * laneCount + laneIndex;

  uint radix = groupId.x;

  uint partitionCount = (elementCount + PARTITION_SIZE - 1) / PARTITION_SIZE;
  printf("elementCount: %u, partitionCount: %u\n", elementCount, partitionCount);
  if (index == 0) {
    reduction = 0;
  }
  GroupMemoryBarrierWithGroupSync();

  for (uint i = 0; WORKGROUP_SIZE * i < partitionCount; ++i) {
    uint partitionIndex = WORKGROUP_SIZE * i + index;
    uint value =
        partitionIndex < partitionCount ? partitionHistogram[RADIX * partitionIndex + radix] : 0;
    uint excl = WavePrefixSum(value) + reduction;
    uint sum = WaveActiveSum(value);

    if (WaveIsFirstLane()) {
      intermediate[waveIndex] = sum;
    }
    GroupMemoryBarrierWithGroupSync();

    if (index < waveCount) {
      uint excl = WavePrefixSum(intermediate[index]);
      uint sum = WaveActiveSum(intermediate[index]);
      intermediate[index] = excl;

      if (index == 0) {
        reduction += sum;
      }
    }
    GroupMemoryBarrierWithGroupSync();

    if (partitionIndex < partitionCount) {
      excl += intermediate[waveIndex];
      partitionHistogram[RADIX * partitionIndex + radix] = excl;
    }
    GroupMemoryBarrierWithGroupSync();
  }

  if (radix == 0)
  {
      bool active = (index < RADIX);
      uint value = 0;
      uint excl0 = 0;
      uint sum0 = 0;

      if (active)
      {
          value = globalHistogram[RADIX * pass + index];
          excl0 = WavePrefixSum(value);
          sum0 = WaveActiveSum(value);

          if (WaveIsFirstLane())
              intermediate[waveIndex] = sum0;
      }

      GroupMemoryBarrierWithGroupSync();

      if (index < RADIX / laneCount)
      {
          uint t = intermediate[index];
          intermediate[index] = WavePrefixSum(t);
      }

      GroupMemoryBarrierWithGroupSync();

      if (active)
      {
          excl0 += intermediate[waveIndex];
          globalHistogram[RADIX * pass + index] = excl0;
      }
  }
}
