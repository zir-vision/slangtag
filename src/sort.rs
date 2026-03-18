use crate::gpu::{BufferMemory, ComputePipeline, DescriptorBuffer, GpuBuffer, GpuQueryPool};
use crate::{ComputeDevice, compute_shader_path, include_u32};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct PassPushConstants {
    pass: u32,
}

struct RadixSortPipelines {
    upsweep: Arc<ComputePipeline>,
    spine: Arc<ComputePipeline>,
    downsweep: Arc<ComputePipeline>,
    downsweep_key_value: Arc<ComputePipeline>,
}

impl RadixSortPipelines {
    fn new(device: &ComputeDevice) -> Self {
        Self {
            upsweep: Arc::new(device.create_compute_pipeline(
                include_u32!(compute_shader_path!("radix/upsweep")),
                Some(32),
            )),
            spine: Arc::new(device.create_compute_pipeline(
                include_u32!(compute_shader_path!("radix/spine")),
                Some(32),
            )),
            downsweep: Arc::new(device.create_compute_pipeline(
                include_u32!(compute_shader_path!("radix/downsweep")),
                Some(32),
            )),
            downsweep_key_value: Arc::new(device.create_compute_pipeline(
                include_u32!(compute_shader_path!("radix/downsweep_key_value")),
                Some(32),
            )),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RadixSorterStorageRequirements {
    pub size: vk::DeviceSize,
    pub usage: vk::BufferUsageFlags,
}

pub struct RadixSorter {
    device: ComputeDevice,
    pipelines: RadixSortPipelines,
}

impl RadixSorter {
    const RADIX: u32 = 256;
    const WORKGROUP_SIZE: u32 = 512;
    const PARTITION_DIVISION: u32 = 8;
    const PARTITION_SIZE: u32 = Self::WORKGROUP_SIZE * Self::PARTITION_DIVISION;
    const PASSES: u32 = 4;

    const BINDING_ELEMENT_COUNTS: u32 = 0;
    const BINDING_GLOBAL_HISTOGRAM: u32 = 1;
    const BINDING_PARTITION_HISTOGRAM: u32 = 2;
    const BINDING_KEYS_IN: u32 = 3;
    const BINDING_KEYS_OUT: u32 = 4;
    const BINDING_VALUES_IN: u32 = 5;
    const BINDING_VALUES_OUT: u32 = 6;

    pub const TIMESTAMP_QUERY_COUNT: u32 = 15;

    pub fn new(device: ComputeDevice) -> Self {
        let pipelines = RadixSortPipelines::new(&device);
        Self { device, pipelines }
    }

    pub fn get_storage_requirements(
        &self,
        max_element_count: u32,
    ) -> RadixSorterStorageRequirements {
        let align = self.required_alignment();
        let element_count_size = Self::align(std::mem::size_of::<u32>() as vk::DeviceSize, align);
        let histogram_size = Self::histogram_size(max_element_count, align);
        let inout_size = Self::inout_size(max_element_count, align);

        let histogram_offset = element_count_size;
        let inout_offset = histogram_offset + histogram_size;
        let storage_size = inout_offset + inout_size;

        RadixSorterStorageRequirements {
            size: storage_size,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        }
    }

    pub fn get_key_value_storage_requirements(
        &self,
        max_element_count: u32,
    ) -> RadixSorterStorageRequirements {
        let align = self.required_alignment();
        let element_count_size = Self::align(std::mem::size_of::<u32>() as vk::DeviceSize, align);
        let histogram_size = Self::histogram_size(max_element_count, align);
        let inout_size = Self::inout_size(max_element_count, align);

        let histogram_offset = element_count_size;
        let inout_offset = histogram_offset + histogram_size;
        let storage_size = inout_offset + Self::align(inout_size, align) + inout_size;

        RadixSorterStorageRequirements {
            size: storage_size,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        }
    }

    pub fn create_storage_buffer(&self, max_element_count: u32) -> GpuBuffer<u32> {
        self.create_storage_buffer_with_requirements(
            self.get_storage_requirements(max_element_count),
        )
    }

    pub fn create_key_value_storage_buffer(&self, max_element_count: u32) -> GpuBuffer<u32> {
        self.create_storage_buffer_with_requirements(
            self.get_key_value_storage_requirements(max_element_count),
        )
    }

    pub fn sort_u32(&self, keys: &[u32]) -> Vec<u32> {
        if keys.is_empty() {
            return Vec::new();
        }

        let gpu_keys = self.upload_u32_storage_buffer(keys);
        let storage = self.create_storage_buffer(keys.len() as u32);
        self.cmd_sort(keys.len() as u32, &gpu_keys, 0, &storage, 0);
        self.download_u32_buffer(&gpu_keys, keys.len())
    }

    pub fn sort_u32_do(&self, keys: &GpuBuffer<u32>, element_count: u32) -> Option<GpuBuffer<u32>> {
        if element_count <= 1 {
            return None;
        }

        let storage = self.create_storage_buffer(element_count);
        self.cmd_sort(element_count, keys, 0, &storage, 0);
        Some(keys.clone())
    }

    pub fn cmd_sort(
        &self,
        element_count: u32,
        keys_buffer: &GpuBuffer<u32>,
        keys_offset: vk::DeviceSize,
        storage_buffer: &GpuBuffer<u32>,
        storage_offset: vk::DeviceSize,
    ) {
        self.cmd_sort_with_query_pool(
            element_count,
            keys_buffer,
            keys_offset,
            storage_buffer,
            storage_offset,
            None,
        );
    }

    pub fn cmd_sort_with_query_pool(
        &self,
        element_count: u32,
        keys_buffer: &GpuBuffer<u32>,
        keys_offset: vk::DeviceSize,
        storage_buffer: &GpuBuffer<u32>,
        storage_offset: vk::DeviceSize,
        query_pool: Option<(&GpuQueryPool, u32)>,
    ) {
        if element_count <= 1 {
            return;
        }

        self.gpu_sort(
            element_count,
            Some(element_count),
            None,
            keys_buffer,
            keys_offset,
            None,
            storage_buffer,
            storage_offset,
            query_pool,
        );
    }

    pub fn cmd_sort_indirect(
        &self,
        max_element_count: u32,
        indirect_buffer: &GpuBuffer<u32>,
        indirect_offset: vk::DeviceSize,
        keys_buffer: &GpuBuffer<u32>,
        keys_offset: vk::DeviceSize,
        storage_buffer: &GpuBuffer<u32>,
        storage_offset: vk::DeviceSize,
    ) {
        self.cmd_sort_indirect_with_query_pool(
            max_element_count,
            indirect_buffer,
            indirect_offset,
            keys_buffer,
            keys_offset,
            storage_buffer,
            storage_offset,
            None,
        );
    }

    pub fn cmd_sort_indirect_with_query_pool(
        &self,
        max_element_count: u32,
        indirect_buffer: &GpuBuffer<u32>,
        indirect_offset: vk::DeviceSize,
        keys_buffer: &GpuBuffer<u32>,
        keys_offset: vk::DeviceSize,
        storage_buffer: &GpuBuffer<u32>,
        storage_offset: vk::DeviceSize,
        query_pool: Option<(&GpuQueryPool, u32)>,
    ) {
        if max_element_count <= 1 {
            return;
        }

        self.gpu_sort(
            max_element_count,
            None,
            Some((indirect_buffer, indirect_offset)),
            keys_buffer,
            keys_offset,
            None,
            storage_buffer,
            storage_offset,
            query_pool,
        );
    }

    pub fn cmd_sort_key_value(
        &self,
        element_count: u32,
        keys_buffer: &GpuBuffer<u32>,
        keys_offset: vk::DeviceSize,
        values_buffer: &GpuBuffer<u32>,
        values_offset: vk::DeviceSize,
        storage_buffer: &GpuBuffer<u32>,
        storage_offset: vk::DeviceSize,
    ) {
        self.cmd_sort_key_value_with_query_pool(
            element_count,
            keys_buffer,
            keys_offset,
            values_buffer,
            values_offset,
            storage_buffer,
            storage_offset,
            None,
        );
    }

    pub fn cmd_sort_key_value_with_query_pool(
        &self,
        element_count: u32,
        keys_buffer: &GpuBuffer<u32>,
        keys_offset: vk::DeviceSize,
        values_buffer: &GpuBuffer<u32>,
        values_offset: vk::DeviceSize,
        storage_buffer: &GpuBuffer<u32>,
        storage_offset: vk::DeviceSize,
        query_pool: Option<(&GpuQueryPool, u32)>,
    ) {
        if element_count <= 1 {
            return;
        }

        self.gpu_sort(
            element_count,
            Some(element_count),
            None,
            keys_buffer,
            keys_offset,
            Some((values_buffer, values_offset)),
            storage_buffer,
            storage_offset,
            query_pool,
        );
    }

    pub fn cmd_sort_key_value_indirect(
        &self,
        max_element_count: u32,
        indirect_buffer: &GpuBuffer<u32>,
        indirect_offset: vk::DeviceSize,
        keys_buffer: &GpuBuffer<u32>,
        keys_offset: vk::DeviceSize,
        values_buffer: &GpuBuffer<u32>,
        values_offset: vk::DeviceSize,
        storage_buffer: &GpuBuffer<u32>,
        storage_offset: vk::DeviceSize,
    ) {
        self.cmd_sort_key_value_indirect_with_query_pool(
            max_element_count,
            indirect_buffer,
            indirect_offset,
            keys_buffer,
            keys_offset,
            values_buffer,
            values_offset,
            storage_buffer,
            storage_offset,
            None,
        );
    }

    pub fn cmd_sort_key_value_indirect_with_query_pool(
        &self,
        max_element_count: u32,
        indirect_buffer: &GpuBuffer<u32>,
        indirect_offset: vk::DeviceSize,
        keys_buffer: &GpuBuffer<u32>,
        keys_offset: vk::DeviceSize,
        values_buffer: &GpuBuffer<u32>,
        values_offset: vk::DeviceSize,
        storage_buffer: &GpuBuffer<u32>,
        storage_offset: vk::DeviceSize,
        query_pool: Option<(&GpuQueryPool, u32)>,
    ) {
        if max_element_count <= 1 {
            return;
        }

        self.gpu_sort(
            max_element_count,
            None,
            Some((indirect_buffer, indirect_offset)),
            keys_buffer,
            keys_offset,
            Some((values_buffer, values_offset)),
            storage_buffer,
            storage_offset,
            query_pool,
        );
    }

    fn gpu_sort(
        &self,
        max_element_count: u32,
        direct_element_count: Option<u32>,
        indirect_count: Option<(&GpuBuffer<u32>, vk::DeviceSize)>,
        keys_buffer: &GpuBuffer<u32>,
        keys_offset: vk::DeviceSize,
        values_buffer: Option<(&GpuBuffer<u32>, vk::DeviceSize)>,
        storage_buffer: &GpuBuffer<u32>,
        storage_offset: vk::DeviceSize,
        query_pool: Option<(&GpuQueryPool, u32)>,
    ) {
        assert!(
            direct_element_count.is_some() ^ indirect_count.is_some(),
            "exactly one of direct or indirect element count must be provided"
        );
        assert!(max_element_count > 0, "max_element_count must be non-zero");

        let align = self.required_alignment();
        Self::assert_alignment(keys_offset, align, "keys_offset");
        Self::assert_alignment(storage_offset, align, "storage_offset");

        if let Some((_, values_offset)) = values_buffer {
            Self::assert_alignment(values_offset, align, "values_offset");
        }

        let element_count_size = Self::align(std::mem::size_of::<u32>() as vk::DeviceSize, align);
        let histogram_size = Self::histogram_size(max_element_count, align);
        let inout_size = Self::inout_size(max_element_count, align);
        let element_bytes =
            (max_element_count as vk::DeviceSize) * (std::mem::size_of::<u32>() as vk::DeviceSize);

        let element_count_offset = storage_offset;
        let histogram_offset = element_count_offset + element_count_size;
        let inout_offset = histogram_offset + histogram_size;

        let global_histogram_size =
            (Self::PASSES as vk::DeviceSize) * (Self::RADIX as vk::DeviceSize) * 4;
        let partition_histogram_offset = histogram_offset + global_histogram_size;
        let partition_count = Self::round_up(max_element_count, Self::PARTITION_SIZE);
        let partition_histogram_size =
            (partition_count as vk::DeviceSize) * (Self::RADIX as vk::DeviceSize) * 4;

        let value_inout_offset = inout_offset + Self::align(inout_size, align);

        let storage_requirements = if values_buffer.is_some() {
            self.get_key_value_storage_requirements(max_element_count)
        } else {
            self.get_storage_requirements(max_element_count)
        };

        Self::assert_range(
            storage_buffer.byte_size(),
            storage_offset,
            storage_requirements.size,
            "storage_buffer",
        );
        Self::assert_range(
            keys_buffer.byte_size(),
            keys_offset,
            element_bytes,
            "keys_buffer",
        );

        if let Some((values, values_offset)) = values_buffer {
            Self::assert_range(
                values.byte_size(),
                values_offset,
                element_bytes,
                "values_buffer",
            );
        }

        self.device.run_commands(|commands| {
            if let Some((query_pool, query)) = query_pool {
                commands.reset_query_pool(query_pool, query, Self::TIMESTAMP_QUERY_COUNT);
                commands.write_timestamp(vk::PipelineStageFlags2::ALL_COMMANDS, query_pool, query + 0);
            }

            if let Some(element_count) = direct_element_count {
                assert!(
                    element_count <= max_element_count,
                    "element_count ({element_count}) exceeds max_element_count ({max_element_count})"
                );
                commands.update_buffer_u32(storage_buffer, element_count_offset, element_count);
            }

            if let Some((indirect_buffer, indirect_offset)) = indirect_count {
                Self::assert_alignment(
                    indirect_offset,
                    std::mem::size_of::<u32>() as vk::DeviceSize,
                    "indirect_offset",
                );
                Self::assert_range(
                    indirect_buffer.byte_size(),
                    indirect_offset,
                    std::mem::size_of::<u32>() as vk::DeviceSize,
                    "indirect_buffer",
                );
                commands.copy_buffer_region(
                    indirect_buffer,
                    indirect_offset,
                    storage_buffer,
                    element_count_offset,
                    std::mem::size_of::<u32>() as vk::DeviceSize,
                );
            }

            commands.fill_buffer_u32_range(storage_buffer, histogram_offset, global_histogram_size, 0);
            commands.barrier_transfer_write_to_compute_read();

            if let Some((query_pool, query)) = query_pool {
                commands.write_timestamp(vk::PipelineStageFlags2::TRANSFER, query_pool, query + 1);
            }

            for pass in 0..Self::PASSES {
                let mut keys_in = keys_buffer.descriptor_range(keys_offset, element_bytes);
                let mut keys_out = storage_buffer.descriptor_range(inout_offset, inout_size);

                let mut values_in_out: Option<(DescriptorBuffer, DescriptorBuffer)> =
                    values_buffer.map(|(values, values_offset)| {
                        (
                            values.descriptor_range(values_offset, element_bytes),
                            storage_buffer.descriptor_range(value_inout_offset, inout_size),
                        )
                    });

                if pass % 2 == 1 {
                    std::mem::swap(&mut keys_in, &mut keys_out);
                    if let Some((values_in, values_out)) = values_in_out.as_mut() {
                        std::mem::swap(values_in, values_out);
                    }
                }

                let mut descriptor_bindings = vec![
                    (
                        Self::BINDING_ELEMENT_COUNTS,
                        storage_buffer.descriptor_range(
                            element_count_offset,
                            std::mem::size_of::<u32>() as vk::DeviceSize,
                        ),
                    ),
                    (
                        Self::BINDING_GLOBAL_HISTOGRAM,
                        storage_buffer.descriptor_range(histogram_offset, global_histogram_size),
                    ),
                    (
                        Self::BINDING_PARTITION_HISTOGRAM,
                        storage_buffer
                            .descriptor_range(partition_histogram_offset, partition_histogram_size),
                    ),
                    (Self::BINDING_KEYS_IN, keys_in),
                    (Self::BINDING_KEYS_OUT, keys_out),
                ];

                if let Some((values_in, values_out)) = values_in_out {
                    descriptor_bindings.push((Self::BINDING_VALUES_IN, values_in));
                    descriptor_bindings.push((Self::BINDING_VALUES_OUT, values_out));
                }

                commands.dispatch_with_push_constants(
                    self.pipelines.upsweep.as_ref(),
                    &descriptor_bindings,
                    &PassPushConstants { pass },
                    [partition_count, 1, 1],
                );
                if let Some((query_pool, query)) = query_pool {
                    commands.write_timestamp(
                        vk::PipelineStageFlags2::COMPUTE_SHADER,
                        query_pool,
                        query + 2 + 3 * pass + 0,
                    );
                }
                commands.barrier_shader_write_to_shader_read();

                commands.dispatch_with_push_constants(
                    self.pipelines.spine.as_ref(),
                    &descriptor_bindings,
                    &PassPushConstants { pass },
                    [Self::RADIX, 1, 1],
                );
                if let Some((query_pool, query)) = query_pool {
                    commands.write_timestamp(
                        vk::PipelineStageFlags2::COMPUTE_SHADER,
                        query_pool,
                        query + 2 + 3 * pass + 1,
                    );
                }
                commands.barrier_shader_write_to_shader_read();

                let downsweep_pipeline = if values_buffer.is_some() {
                    self.pipelines.downsweep_key_value.as_ref()
                } else {
                    self.pipelines.downsweep.as_ref()
                };
                commands.dispatch_with_push_constants(
                    downsweep_pipeline,
                    &descriptor_bindings,
                    &PassPushConstants { pass },
                    [partition_count, 1, 1],
                );
                if let Some((query_pool, query)) = query_pool {
                    commands.write_timestamp(
                        vk::PipelineStageFlags2::COMPUTE_SHADER,
                        query_pool,
                        query + 2 + 3 * pass + 2,
                    );
                }

                if pass < (Self::PASSES - 1) {
                    commands.barrier_shader_write_to_shader_read();
                }
            }

            if let Some((query_pool, query)) = query_pool {
                commands.write_timestamp(vk::PipelineStageFlags2::ALL_COMMANDS, query_pool, query + 14);
            }
        });
    }

    fn required_alignment(&self) -> vk::DeviceSize {
        self.device
            .min_storage_buffer_offset_alignment()
            .max(std::mem::size_of::<u32>() as vk::DeviceSize)
    }

    fn round_up(a: u32, b: u32) -> u32 {
        a.div_ceil(b)
    }

    fn align(a: vk::DeviceSize, b: vk::DeviceSize) -> vk::DeviceSize {
        a.div_ceil(b) * b
    }

    fn histogram_size(element_count: u32, align: vk::DeviceSize) -> vk::DeviceSize {
        let partition_count = Self::round_up(element_count, Self::PARTITION_SIZE);
        let words = 4u64
            + (Self::PASSES as u64) * (Self::RADIX as u64)
            + (partition_count as u64) * (Self::RADIX as u64);
        Self::align(
            words * (std::mem::size_of::<u32>() as vk::DeviceSize),
            align,
        )
    }

    fn inout_size(element_count: u32, align: vk::DeviceSize) -> vk::DeviceSize {
        let bytes =
            (element_count as vk::DeviceSize) * (std::mem::size_of::<u32>() as vk::DeviceSize);
        Self::align(bytes, align)
    }

    fn assert_alignment(offset: vk::DeviceSize, align: vk::DeviceSize, label: &str) {
        assert!(
            offset % align == 0,
            "{label} ({offset}) must be aligned to {align} bytes"
        );
    }

    fn assert_range(
        buffer_size: vk::DeviceSize,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
        label: &str,
    ) {
        let end = offset
            .checked_add(size)
            .unwrap_or_else(|| panic!("{label} range overflow: offset={offset}, size={size}"));
        assert!(
            end <= buffer_size,
            "{label} range out of bounds: offset={offset}, size={size}, buffer_size={buffer_size}"
        );
    }

    fn create_storage_buffer_with_requirements(
        &self,
        requirements: RadixSorterStorageRequirements,
    ) -> GpuBuffer<u32> {
        assert!(
            requirements.size % (std::mem::size_of::<u32>() as vk::DeviceSize) == 0,
            "storage requirement size ({}) is not aligned to u32",
            requirements.size
        );

        let word_len = usize::try_from(requirements.size / (std::mem::size_of::<u32>() as u64))
            .expect("storage requirement size does not fit in usize");

        self.device
            .create_buffer(word_len, requirements.usage, BufferMemory::DeviceLocal)
    }

    fn upload_u32_storage_buffer(&self, data: &[u32]) -> GpuBuffer<u32> {
        self.device.upload_buffer(
            data,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            true,
        )
    }

    fn download_u32_buffer(&self, source: &GpuBuffer<u32>, len: usize) -> Vec<u32> {
        let destination = self.device.create_buffer(
            len,
            vk::BufferUsageFlags::TRANSFER_DST,
            BufferMemory::HostRandomAccess,
        );
        self.device
            .copy_buffer(source, &destination, destination.byte_size());
        destination.read(len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn maybe_sorter() -> Option<RadixSorter> {
        std::panic::catch_unwind(|| {
            let device = ComputeDevice::new_default();
            RadixSorter::new(device)
        })
        .ok()
    }

    fn assert_gpu_sort_matches_cpu(input: &[u32]) {
        let _guard = test_lock().lock().expect("test lock poisoned");

        let Some(sorter) = maybe_sorter() else {
            panic!("failed to create RadixSorter, likely due to Vulkan initialization failure");
        };

        let mut expected = input.to_vec();
        expected.sort_unstable();
        let actual = sorter.sort_u32(input);
        assert_eq!(actual, expected);
    }

    #[test]
    fn sort_u32_small_unsorted() {
        assert_gpu_sort_matches_cpu(&[9, 1, 5, 3, 7, 0, 2, 4, 8, 6]);
    }
}
