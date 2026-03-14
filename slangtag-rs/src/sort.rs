use crate::{ComputeDevice, compute_shader_path};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use std::collections::BTreeMap;
use std::sync::Arc;
use vulkano::{NonExhaustive, VulkanObject};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferBeginInfo, CommandBufferLevel, CommandBufferUsage,
    CopyBufferInfo, RecordingCommandBuffer,
};
use vulkano::descriptor_set::{
    DescriptorSet, WriteDescriptorSet,
    layout::{
        DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
        DescriptorType,
    },
};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::Pipeline;
use vulkano::pipeline::{
    ComputePipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    compute::ComputePipelineCreateInfo,
    layout::{PipelineLayoutCreateInfo, PushConstantRange},
};
use vulkano::shader::{ShaderModuleCreateInfo, ShaderStages, spirv::bytes_to_words};
use vulkano::sync::{self, AccessFlags, DependencyInfo, GpuFuture, MemoryBarrier, PipelineStages};

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct PassPushConstants {
    pass: i32,
}

struct RadixSortPipelines {
    upsweep: Arc<ComputePipeline>,
    spine: Arc<ComputePipeline>,
    downsweep: Arc<ComputePipeline>,
}

impl RadixSortPipelines {
    fn new(
        device: &ComputeDevice,
        layout: &Arc<PipelineLayout>,
        required_subgroup_size: u32,
    ) -> Self {
        Self {
            upsweep: RadixSorter::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!("radix/upsweep")),
                layout.clone(),
                required_subgroup_size,
            ),
            spine: RadixSorter::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!("radix/spine")),
                layout.clone(),
                required_subgroup_size,
            ),
            downsweep: RadixSorter::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!("radix/downsweep")),
                layout.clone(),
                required_subgroup_size,
            ),
        }
    }
}

pub struct RadixSorter {
    device: ComputeDevice,
    pipeline_layout: Arc<PipelineLayout>,
    pipelines: RadixSortPipelines,
}

impl RadixSorter {
    const RADIX: u32 = 256;
    const WORKGROUP_SIZE: u32 = 512;
    const PARTITION_DIVISION: u32 = 8;
    const PARTITION_SIZE: u32 = Self::WORKGROUP_SIZE * Self::PARTITION_DIVISION;
    const PASSES: u32 = 4;
    const REQUIRED_SUBGROUP_SIZE: u32 = 32;
    const BINDING_ELEMENT_COUNTS: u32 = 0;
    const BINDING_GLOBAL_HISTOGRAM: u32 = 1;
    const BINDING_PARTITION_HISTOGRAM: u32 = 2;
    const BINDING_KEYS_IN: u32 = 3;
    const BINDING_KEYS_OUT: u32 = 4;
    const BINDING_VALUES_IN: u32 = 5;
    const BINDING_VALUES_OUT: u32 = 6;    

    pub fn new(device: ComputeDevice) -> Self {
        let required_subgroup_size = Self::required_subgroup_size(&device);
        let pipeline_layout = Self::create_pipeline_layout(&device);
        let pipelines = RadixSortPipelines::new(&device, &pipeline_layout, required_subgroup_size);
        Self {
            device,
            pipeline_layout,
            pipelines,
        }
    }

    pub fn sort_u32(&self, keys: &[u32]) -> Vec<u32> {
        if keys.is_empty() {
            return Vec::new();
        }

        let gpu_keys = self.upload_u32_storage_buffer(keys);
        let keys_out = self.sort_u32_do(&gpu_keys, keys.len() as u32);
        self.download_u32_buffer(&gpu_keys, keys.len())
    }

    pub fn sort_u32_do(&self, keys: &Subbuffer<[u32]>, element_count: u32) -> Option<Subbuffer<[u32]>> {
        if element_count <= 1 {
            return None;
        }

        assert!(
            (element_count as u64) <= keys.len() as u64,
            "element_count ({element_count}) exceeds key buffer length ({})",
            keys.len()
        );

        let partition_count = element_count.div_ceil(Self::PARTITION_SIZE);

        let element_counts = Buffer::from_iter(
            self.device.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            std::iter::once(element_count),
        )
        .expect("failed to create element-count buffer");

        

        let global_histogram =
            self.new_zeroed_u32_storage_buffer((Self::RADIX * Self::PASSES) as usize);
        let partition_histogram =
            self.new_zeroed_u32_storage_buffer((Self::RADIX * partition_count) as usize);
        let keys_out = self.new_u32_storage_buffer(keys.len() as usize);
        let mut command_buffer = RecordingCommandBuffer::new(
            self.device.command_buffer_allocator.clone(),
            self.device.queue_family_index,
            CommandBufferLevel::Primary,
            CommandBufferBeginInfo {
                usage: CommandBufferUsage::OneTimeSubmit,
                ..Default::default()
            },
        )
        .expect("failed to create recording command buffer");


        for pass in 0..(Self::PASSES as i32) {
            let mut descriptor_writes = vec![
                WriteDescriptorSet::buffer(Self::BINDING_ELEMENT_COUNTS, element_counts.clone()),
                WriteDescriptorSet::buffer(
                    Self::BINDING_GLOBAL_HISTOGRAM,
                    global_histogram.clone(),
                ),
                WriteDescriptorSet::buffer(
                    Self::BINDING_PARTITION_HISTOGRAM,
                    partition_histogram.clone(),
                ),
                WriteDescriptorSet::buffer(Self::BINDING_KEYS_IN, keys.clone()),
                WriteDescriptorSet::buffer(Self::BINDING_KEYS_OUT, keys_out.clone()),
            ];
            if pass % 2 == 1 {
                descriptor_writes.swap(3, 4);
            }
            unsafe {
                command_buffer
                    .push_descriptor_set(
                        vulkano::pipeline::PipelineBindPoint::Compute,
                        &self.pipeline_layout.clone(),
                        0,
                        &descriptor_writes,
                    )
                    .expect("failed to push descriptor set");
                command_buffer
                    .push_constants(
                        &self.pipeline_layout.clone(),
                        0,
                        &PassPushConstants { pass },
                    )
                    .expect("failed to push constants");
                command_buffer
                    .bind_pipeline_compute(&self.pipelines.upsweep.clone())
                    .expect("failed to bind upsweep pipeline");
                command_buffer
                    .dispatch([partition_count, 1, 1])
                    .expect("failed to dispatch upsweep");
                command_buffer
                    .pipeline_barrier(&Self::shader_write_read_barrier())
                    .expect("failed to record pipeline barrier after upsweep");
                command_buffer
                    .bind_pipeline_compute(&self.pipelines.spine.clone())
                    .expect("failed to bind spine pipeline");
                command_buffer
                    .dispatch([Self::RADIX, 1, 1])
                    .expect("failed to dispatch spine");
                command_buffer
                    .pipeline_barrier(&Self::shader_write_read_barrier())
                    .expect("failed to record pipeline barrier after spine");
                command_buffer
                    .bind_pipeline_compute(&self.pipelines.downsweep.clone())
                    .expect("failed to bind downsweep pipeline");
                command_buffer
                    .dispatch([partition_count, 1, 1])
                    .expect("failed to dispatch downsweep");
                if pass < 3 {
                    command_buffer
                        .pipeline_barrier(&Self::shader_write_read_barrier())
                        .expect("failed to record pipeline barrier after downsweep");
                }

            }
        }

        let command_buffer = unsafe { command_buffer.end() }.expect("failed to end command buffer");
        let command_buffer_handle = vec![command_buffer.handle()];
        let submit_info = ash::vk::SubmitInfo::default().command_buffers(&command_buffer_handle);
        let fence = vulkano::sync::fence::Fence::new(
            self.device.device.clone(),
            vulkano::sync::fence::FenceCreateInfo {
                ..Default::default()
            },
        )
        .unwrap();
        
        self.device
            .queue
            .with(|guard| unsafe {
                (self.device.device.fns().v1_0.queue_submit)(
                    self.device.queue.handle(),
                    1,
                    &raw const submit_info,
                    fence.handle(),
                )
                .result()
                .unwrap();
            });
        fence.wait(None).expect("failed to wait for command buffer execution");

        Some(keys_out)
    }

    fn create_compute_pipeline(
        device: &ComputeDevice,
        module_bytes: &[u8],
        layout: Arc<PipelineLayout>,
        required_subgroup_size: u32,
    ) -> Arc<ComputePipeline> {
        let shader = unsafe {
            vulkano::shader::ShaderModule::new(
                device.device.clone(),
                ShaderModuleCreateInfo::new(&bytes_to_words(module_bytes).unwrap()),
            )
            .expect("failed to create shader module")
        };

        let entry_point = shader
            .entry_point("main")
            .expect("failed to find entry point in shader");

        let mut stage = PipelineShaderStageCreateInfo::new(entry_point);
        stage.required_subgroup_size = Some(required_subgroup_size);

        ComputePipeline::new(
            device.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .expect("failed to create compute pipeline")
    }

    fn required_subgroup_size(device: &ComputeDevice) -> u32 {
        let properties = device.device.physical_device().properties();
        let supported = device.device.enabled_features().subgroup_size_control
            && properties
                .required_subgroup_size_stages
                .unwrap_or_default()
                .intersects(ShaderStages::COMPUTE)
            && properties.min_subgroup_size.unwrap_or(1) <= Self::REQUIRED_SUBGROUP_SIZE
            && properties.max_subgroup_size.unwrap_or(u32::MAX) >= Self::REQUIRED_SUBGROUP_SIZE
            && properties
                .max_compute_workgroup_subgroups
                .unwrap_or(0)
                .saturating_mul(Self::REQUIRED_SUBGROUP_SIZE)
                >= Self::WORKGROUP_SIZE;

        assert!(
            supported,
            "RadixSorter requires a 32-lane compute subgroup. \
             Enable `subgroup_size_control` and run on hardware that supports required subgroup size 32."
        );

        Self::REQUIRED_SUBGROUP_SIZE
    }

    fn shader_write_read_barrier() -> DependencyInfo {
        let mut dependency = DependencyInfo::default();
        dependency.memory_barriers.push(MemoryBarrier {
            src_stages: PipelineStages::COMPUTE_SHADER,
            src_access: AccessFlags::SHADER_WRITE,
            dst_stages: PipelineStages::COMPUTE_SHADER,
            dst_access: AccessFlags::SHADER_READ,
            ..Default::default()
        });
        dependency
    }

    fn create_pipeline_layout(device: &ComputeDevice) -> Arc<PipelineLayout> {
        let mut bindings = BTreeMap::new();
        for binding in [
            Self::BINDING_ELEMENT_COUNTS,
            Self::BINDING_GLOBAL_HISTOGRAM,
            Self::BINDING_PARTITION_HISTOGRAM,
            Self::BINDING_KEYS_IN,
            Self::BINDING_KEYS_OUT,
            // Self::BINDING_VALUES_IN,
            // Self::BINDING_VALUES_OUT,
        ] {
            let mut layout_binding =
                DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer);
            layout_binding.stages = ShaderStages::COMPUTE;
            layout_binding.descriptor_count = 1;
            bindings.insert(binding, layout_binding);
        }

        let descriptor_set_layout = DescriptorSetLayout::new(
            device.device.clone(),
            DescriptorSetLayoutCreateInfo {
                bindings,
                flags: vulkano::descriptor_set::layout::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR,
                ..Default::default()
            },
        )
        .expect("failed to create radix descriptor set layout");

        PipelineLayout::new(
            device.device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![descriptor_set_layout],
                push_constant_ranges: vec![PushConstantRange {
                    stages: ShaderStages::COMPUTE,
                    offset: 0,
                    size: std::mem::size_of::<PassPushConstants>() as u32,
                }],
                ..Default::default()
            },
        )
        .expect("failed to create radix pipeline layout")
    }

    fn create_descriptor_set(&self, writes: Vec<WriteDescriptorSet>) -> Arc<DescriptorSet> {
        let descriptor_set_layout = self
            .pipeline_layout
            .set_layouts()
            .first()
            .expect("missing descriptor set layout");

        DescriptorSet::new(
            self.device.descriptor_set_allocator.clone(),
            descriptor_set_layout.clone(),
            writes,
            [],
        )
        .expect("failed to create descriptor set")
    }

    fn new_u32_storage_buffer(&self, len: usize) -> Subbuffer<[u32]> {
        Buffer::new_unsized(
            self.device.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            len as u64,
        )
        .expect("failed to create u32 storage buffer")
    }

    fn new_zeroed_u32_storage_buffer(&self, len: usize) -> Subbuffer<[u32]> {
        let buf = Buffer::new_unsized(
            self.device.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            len as u64,
        )
        .expect("failed to create zeroed u32 storage buffer");

        let mut builder = AutoCommandBufferBuilder::primary(
            self.device.command_buffer_allocator.clone(),
            self.device.queue_family_index,
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("failed to create command buffer builder");
        builder
            .fill_buffer(buf.clone(), 0)
            .expect("failed to record fill buffer command");

        let command_buffer = builder.build().expect("failed to build command buffer");
        let future = sync::now(self.device.device.clone())
            .then_execute(self.device.queue.clone(), command_buffer)
            .expect("failed to execute command buffer")
            .then_signal_fence_and_flush()
            .expect("failed to flush command buffer");
        future.wait(None).expect("failed to wait for buffer fill");

        buf
    }

    fn upload_u32_storage_buffer(&self, data: &[u32]) -> Subbuffer<[u32]> {
        Buffer::from_iter(
            self.device.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data.iter().copied(),
        )
        .expect("failed to upload u32 storage buffer")
    }

    fn download_u32_buffer(&self, source: &Subbuffer<[u32]>, len: usize) -> Vec<u32> {
        let destination: Subbuffer<[u32]> = Buffer::new_unsized(
            self.device.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            len as u64,
        )
        .expect("failed to create destination buffer");

        self.copy_u32_buffer(source, &destination);

        destination
            .read()
            .expect("failed to map destination buffer")
            .to_vec()
    }

    fn copy_u32_buffer(&self, src: &Subbuffer<[u32]>, dst: &Subbuffer<[u32]>) {
        let mut builder = AutoCommandBufferBuilder::primary(
            self.device.command_buffer_allocator.clone(),
            self.device.queue_family_index,
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("failed to create command buffer builder");

        builder
            .copy_buffer(CopyBufferInfo::buffers(src.clone(), dst.clone()))
            .expect("failed to record copy buffer command");

        let command_buffer = builder.build().expect("failed to build command buffer");
        let future = sync::now(self.device.device.clone())
            .then_execute(self.device.queue.clone(), command_buffer)
            .expect("failed to execute command buffer")
            .then_signal_fence_and_flush()
            .expect("failed to flush command buffer");
        future.wait(None).expect("failed to wait for buffer copy");
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use std::{
        sync::{Mutex, OnceLock},
    };

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
