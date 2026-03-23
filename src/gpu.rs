use ash::vk;
use bytemuck::Pod;
use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ptr;
use std::sync::{Arc, Mutex};
use vk_mem::{
    Alloc, Allocation, AllocationCreateFlags, AllocationCreateInfo, Allocator, MemoryUsage,
};

const MAX_STORAGE_BINDINGS: u32 = 8;
const MAX_PUSH_CONSTANT_BYTES: u32 = 128;
const DESCRIPTOR_SET_CAPACITY: u32 = 1024;

#[derive(Clone)]
pub struct ComputeDevice {
    inner: Arc<ComputeDeviceInner>,
}

pub struct ComputeCommandContext {
    inner: Arc<ComputeDeviceInner>,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    submit_fence: vk::Fence,
}

struct ComputeDeviceInner {
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    queue_family_index: u32,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    allocator: ManuallyDrop<Allocator>,
    submit_lock: Mutex<()>,
    descriptor_pool_lock: Mutex<()>,
    subgroup_size_control_enabled: bool,
    required_subgroup_size_stages: vk::ShaderStageFlags,
    min_subgroup_size: u32,
    max_subgroup_size: u32,
    max_compute_workgroup_subgroups: u32,
    min_storage_buffer_offset_alignment: vk::DeviceSize,
    timestamp_period_ns: f32,
}

impl Drop for ComputeDeviceInner {
    fn drop(&mut self) {
        let _submit_lock = self
            .submit_lock
            .lock()
            .expect("failed to lock submit path during device drop");
        unsafe {
            let _ = self.device.device_wait_idle();
            ManuallyDrop::drop(&mut self.allocator);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
            let _ = &self.entry;
        }
    }
}

impl Drop for ComputeCommandContext {
    fn drop(&mut self) {
        let _submit_lock = self
            .inner
            .submit_lock
            .lock()
            .expect("failed to lock submit path during command context drop");
        unsafe {
            let _ = self.inner.device.device_wait_idle();
            self.inner.device.destroy_fence(self.submit_fence, None);
            self.inner
                .device
                .free_command_buffers(self.command_pool, &[self.command_buffer]);
            self.inner
                .device
                .destroy_command_pool(self.command_pool, None);
        }
    }
}

#[derive(Clone)]
pub struct ComputePipeline {
    device: Arc<ComputeDeviceInner>,
    pipeline: vk::Pipeline,
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_pipeline(self.pipeline, None);
        }
    }
}

#[derive(Clone)]
pub struct GpuQueryPool {
    device: Arc<ComputeDeviceInner>,
    query_pool: vk::QueryPool,
    query_type: vk::QueryType,
    query_count: u32,
}

impl Drop for GpuQueryPool {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_query_pool(self.query_pool, None);
        }
    }
}

#[derive(Clone)]
pub struct GpuBuffer<T: Pod + Copy> {
    raw: Arc<RawBuffer>,
    len: usize,
    _marker: PhantomData<T>,
}

struct RawBuffer {
    device: Arc<ComputeDeviceInner>,
    buffer: vk::Buffer,
    allocation: Mutex<Allocation>,
    bytes: vk::DeviceSize,
}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        let mut allocation = self
            .allocation
            .lock()
            .expect("failed to lock allocation during buffer drop");
        unsafe {
            self.device
                .allocator
                .destroy_buffer(self.buffer, &mut allocation);
        }
    }
}

#[derive(Clone, Copy)]
pub enum BufferMemory {
    DeviceLocal,
    HostSequentialWrite,
    HostRandomAccess,
}

#[derive(Clone, Copy)]
pub struct DescriptorBuffer {
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    range: vk::DeviceSize,
}

impl DescriptorBuffer {
    pub fn new(buffer: vk::Buffer, offset: vk::DeviceSize, range: vk::DeviceSize) -> Self {
        assert!(
            buffer != vk::Buffer::null(),
            "descriptor buffer must not be null"
        );
        assert!(range > 0, "descriptor range must be greater than zero");
        Self {
            buffer,
            offset,
            range,
        }
    }

    pub fn from_vk_buffer(buffer: vk::Buffer, range: vk::DeviceSize) -> Self {
        Self::new(buffer, 0, range)
    }
}

impl<T: Pod + Copy> From<&GpuBuffer<T>> for DescriptorBuffer {
    fn from(value: &GpuBuffer<T>) -> Self {
        Self {
            buffer: value.raw.buffer,
            offset: 0,
            range: value.raw.bytes,
        }
    }
}

fn main_entry_name() -> &'static CStr {
    c"main"
}

pub(crate) struct CommandRecorder<'a> {
    inner: &'a ComputeDeviceInner,
    command_buffer: vk::CommandBuffer,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl<'a> CommandRecorder<'a> {
    fn allocate_and_write_descriptor_set(
        &mut self,
        bindings: &[(u32, DescriptorBuffer)],
    ) -> vk::DescriptorSet {
        let descriptor_set = {
            let _pool_lock = self
                .inner
                .descriptor_pool_lock
                .lock()
                .expect("failed to lock descriptor pool");
            let set_layouts = [self.inner.descriptor_set_layout];
            let allocate_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.inner.descriptor_pool)
                .set_layouts(&set_layouts);
            unsafe {
                self.inner
                    .device
                    .allocate_descriptor_sets(&allocate_info)
                    .expect("failed to allocate descriptor set")[0]
            }
        };
        self.descriptor_sets.push(descriptor_set);

        let buffer_infos: Vec<_> = bindings
            .iter()
            .map(|(_, binding)| {
                vk::DescriptorBufferInfo::default()
                    .buffer(binding.buffer)
                    .offset(binding.offset)
                    .range(binding.range)
            })
            .collect();
        let writes: Vec<_> = bindings
            .iter()
            .zip(buffer_infos.iter())
            .map(|((binding_index, _), info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(*binding_index)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        unsafe {
            self.inner.device.update_descriptor_sets(&writes, &[]);
        }

        descriptor_set
    }

    pub fn reset_query_pool(
        &mut self,
        query_pool: &GpuQueryPool,
        first_query: u32,
        query_count: u32,
    ) {
        query_pool.assert_range(first_query, query_count);
        unsafe {
            self.inner.device.cmd_reset_query_pool(
                self.command_buffer,
                query_pool.query_pool,
                first_query,
                query_count,
            );
        }
    }

    pub fn write_timestamp(
        &mut self,
        stage: vk::PipelineStageFlags2,
        query_pool: &GpuQueryPool,
        query: u32,
    ) {
        assert!(
            query_pool.query_type == vk::QueryType::TIMESTAMP,
            "write_timestamp requires a TIMESTAMP query pool"
        );
        query_pool.assert_range(query, 1);
        unsafe {
            self.inner.device.cmd_write_timestamp2(
                self.command_buffer,
                stage,
                query_pool.query_pool,
                query,
            );
        }
    }

    pub fn copy_buffer_region<T: Pod + Copy>(
        &mut self,
        src: &GpuBuffer<T>,
        src_offset: vk::DeviceSize,
        dst: &GpuBuffer<T>,
        dst_offset: vk::DeviceSize,
        bytes: vk::DeviceSize,
    ) {
        assert!(
            src_offset.saturating_add(bytes) <= src.raw.bytes,
            "copy source range exceeds buffer: offset={} bytes={} src_size={}",
            src_offset,
            bytes,
            src.raw.bytes
        );
        assert!(
            dst_offset.saturating_add(bytes) <= dst.raw.bytes,
            "copy destination range exceeds buffer: offset={} bytes={} dst_size={}",
            dst_offset,
            bytes,
            dst.raw.bytes
        );

        unsafe {
            let copy_regions = [vk::BufferCopy::default()
                .src_offset(src_offset)
                .dst_offset(dst_offset)
                .size(bytes)];
            self.inner.device.cmd_copy_buffer(
                self.command_buffer,
                src.raw.buffer,
                dst.raw.buffer,
                &copy_regions,
            );
        }
    }

    pub fn fill_buffer_u32_range(
        &mut self,
        buffer: &GpuBuffer<u32>,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
        value: u32,
    ) {
        assert!(
            offset.saturating_add(size) <= buffer.raw.bytes,
            "fill range exceeds buffer: offset={} size={} buffer_size={}",
            offset,
            size,
            buffer.raw.bytes
        );

        unsafe {
            self.inner.device.cmd_fill_buffer(
                self.command_buffer,
                buffer.raw.buffer,
                offset,
                size,
                value,
            );
        }
    }

    pub fn update_buffer_u32(
        &mut self,
        buffer: &GpuBuffer<u32>,
        offset: vk::DeviceSize,
        value: u32,
    ) {
        assert!(
            offset % (std::mem::size_of::<u32>() as vk::DeviceSize) == 0,
            "update offset ({offset}) must be aligned to 4 bytes"
        );
        assert!(
            offset.saturating_add(std::mem::size_of::<u32>() as vk::DeviceSize) <= buffer.raw.bytes,
            "update range exceeds buffer: offset={} size={} buffer_size={}",
            offset,
            std::mem::size_of::<u32>(),
            buffer.raw.bytes
        );

        unsafe {
            self.inner.device.cmd_update_buffer(
                self.command_buffer,
                buffer.raw.buffer,
                offset,
                bytemuck::bytes_of(&value),
            );
        }
    }

    pub fn barrier_transfer_write_to_compute_read(&mut self) {
        let memory_barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)];
        unsafe {
            self.inner.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &memory_barrier,
                &[],
                &[],
            );
        }
    }

    #[allow(dead_code)]
    pub fn barrier_transfer_write_to_indirect_read(&mut self) {
        let memory_barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ)];
        unsafe {
            self.inner.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::DRAW_INDIRECT,
                vk::DependencyFlags::empty(),
                &memory_barrier,
                &[],
                &[],
            );
        }
    }

    pub fn barrier_shader_write_to_shader_read(&mut self) {
        let memory_barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)];
        unsafe {
            self.inner.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &memory_barrier,
                &[],
                &[],
            );
        }
    }

    pub fn barrier_shader_write_to_indirect_read(&mut self) {
        let memory_barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ)];
        unsafe {
            self.inner.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::DRAW_INDIRECT,
                vk::DependencyFlags::empty(),
                &memory_barrier,
                &[],
                &[],
            );
        }
    }

    pub fn barrier_shader_write_to_transfer_read(&mut self) {
        let memory_barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
        unsafe {
            self.inner.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &memory_barrier,
                &[],
                &[],
            );
        }
    }

    pub fn dispatch_with_push_constants<T: Pod + Copy>(
        &mut self,
        pipeline: &ComputePipeline,
        bindings: &[(u32, DescriptorBuffer)],
        push_constants: &T,
        groups: [u32; 3],
    ) {
        let _ = self.dispatch_with_push_constants_capture_set(
            pipeline,
            bindings,
            push_constants,
            groups,
        );
    }

    pub fn dispatch_with_push_constants_capture_set<T: Pod + Copy>(
        &mut self,
        pipeline: &ComputePipeline,
        bindings: &[(u32, DescriptorBuffer)],
        push_constants: &T,
        groups: [u32; 3],
    ) -> vk::DescriptorSet {
        let push_bytes = bytemuck::bytes_of(push_constants);
        assert!(
            (push_bytes.len() as u32) <= MAX_PUSH_CONSTANT_BYTES,
            "push constants are too large ({} bytes > {} bytes)",
            push_bytes.len(),
            MAX_PUSH_CONSTANT_BYTES
        );

        let descriptor_set = self.allocate_and_write_descriptor_set(bindings);
        unsafe {
            self.inner.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.pipeline,
            );
            self.inner.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.inner.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            self.inner.device.cmd_push_constants(
                self.command_buffer,
                self.inner.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_bytes,
            );
            self.inner
                .device
                .cmd_dispatch(self.command_buffer, groups[0], groups[1], groups[2]);
        }

        descriptor_set
    }

    pub fn dispatch_indirect_with_push_constants<T: Pod + Copy>(
        &mut self,
        pipeline: &ComputePipeline,
        bindings: &[(u32, DescriptorBuffer)],
        push_constants: &T,
        indirect_buffer: &GpuBuffer<u32>,
        indirect_offset: vk::DeviceSize,
    ) {
        let _ = self.dispatch_indirect_with_push_constants_capture_set(
            pipeline,
            bindings,
            push_constants,
            indirect_buffer,
            indirect_offset,
        );
    }

    pub fn dispatch_indirect_with_push_constants_capture_set<T: Pod + Copy>(
        &mut self,
        pipeline: &ComputePipeline,
        bindings: &[(u32, DescriptorBuffer)],
        push_constants: &T,
        indirect_buffer: &GpuBuffer<u32>,
        indirect_offset: vk::DeviceSize,
    ) -> vk::DescriptorSet {
        let push_bytes = bytemuck::bytes_of(push_constants);
        assert!(
            (push_bytes.len() as u32) <= MAX_PUSH_CONSTANT_BYTES,
            "push constants are too large ({} bytes > {} bytes)",
            push_bytes.len(),
            MAX_PUSH_CONSTANT_BYTES
        );
        assert!(
            indirect_offset % (std::mem::size_of::<u32>() as vk::DeviceSize) == 0,
            "indirect_offset ({indirect_offset}) must be aligned to 4 bytes"
        );
        assert!(
            indirect_offset.saturating_add((3 * std::mem::size_of::<u32>()) as vk::DeviceSize)
                <= indirect_buffer.raw.bytes,
            "indirect range exceeds buffer: offset={} size={} buffer_size={}",
            indirect_offset,
            3 * std::mem::size_of::<u32>(),
            indirect_buffer.raw.bytes
        );

        let descriptor_set = self.allocate_and_write_descriptor_set(bindings);
        unsafe {
            self.inner.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.pipeline,
            );
            self.inner.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.inner.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            self.inner.device.cmd_push_constants(
                self.command_buffer,
                self.inner.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_bytes,
            );
            self.inner.device.cmd_dispatch_indirect(
                self.command_buffer,
                indirect_buffer.raw.buffer,
                indirect_offset,
            );
        }

        descriptor_set
    }

    #[allow(dead_code)]
    pub fn dispatch_indirect(
        &mut self,
        pipeline: &ComputePipeline,
        bindings: &[(u32, DescriptorBuffer)],
        indirect_buffer: &GpuBuffer<u32>,
        indirect_offset: vk::DeviceSize,
    ) {
        assert!(
            indirect_offset % (std::mem::size_of::<u32>() as vk::DeviceSize) == 0,
            "indirect_offset ({indirect_offset}) must be aligned to 4 bytes"
        );
        assert!(
            indirect_offset.saturating_add((3 * std::mem::size_of::<u32>()) as vk::DeviceSize)
                <= indirect_buffer.raw.bytes,
            "indirect range exceeds buffer: offset={} size={} buffer_size={}",
            indirect_offset,
            3 * std::mem::size_of::<u32>(),
            indirect_buffer.raw.bytes
        );

        let descriptor_set = {
            let _pool_lock = self
                .inner
                .descriptor_pool_lock
                .lock()
                .expect("failed to lock descriptor pool");
            let set_layouts = [self.inner.descriptor_set_layout];
            let allocate_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.inner.descriptor_pool)
                .set_layouts(&set_layouts);
            unsafe {
                self.inner
                    .device
                    .allocate_descriptor_sets(&allocate_info)
                    .expect("failed to allocate descriptor set")[0]
            }
        };
        self.descriptor_sets.push(descriptor_set);

        let buffer_infos: Vec<_> = bindings
            .iter()
            .map(|(_, binding)| {
                vk::DescriptorBufferInfo::default()
                    .buffer(binding.buffer)
                    .offset(binding.offset)
                    .range(binding.range)
            })
            .collect();
        let writes: Vec<_> = bindings
            .iter()
            .zip(buffer_infos.iter())
            .map(|((binding_index, _), info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(*binding_index)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        unsafe {
            self.inner.device.update_descriptor_sets(&writes, &[]);
            self.inner.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.pipeline,
            );
            self.inner.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.inner.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            self.inner.device.cmd_dispatch_indirect(
                self.command_buffer,
                indirect_buffer.raw.buffer,
                indirect_offset,
            );
        }
    }
}

pub(crate) struct CachedCommandBuffer {
    inner: Arc<ComputeDeviceInner>,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    submit_fence: vk::Fence,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl Drop for CachedCommandBuffer {
    fn drop(&mut self) {
        let _submit_lock = self
            .inner
            .submit_lock
            .lock()
            .expect("failed to lock submit path during cached command buffer drop");
        if !self.descriptor_sets.is_empty() {
            let _pool_lock = self
                .inner
                .descriptor_pool_lock
                .lock()
                .expect("failed to lock descriptor pool during cached command buffer drop");
            unsafe {
                let _ = self
                    .inner
                    .device
                    .free_descriptor_sets(self.inner.descriptor_pool, &self.descriptor_sets);
            }
        }
        unsafe {
            let _ = self.inner.device.device_wait_idle();
            self.inner.device.destroy_fence(self.submit_fence, None);
            self.inner
                .device
                .free_command_buffers(self.command_pool, &[self.command_buffer]);
            self.inner
                .device
                .destroy_command_pool(self.command_pool, None);
        }
    }
}

impl CachedCommandBuffer {
    pub(crate) fn record<F>(&mut self, record: F)
    where
        F: FnOnce(&mut CommandRecorder<'_>),
    {
        let _submit_lock = self
            .inner
            .submit_lock
            .lock()
            .expect("failed to lock submit path for command recording");
        unsafe {
            self.inner
                .device
                .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())
                .expect("failed to reset cached command pool");
            self.inner
                .device
                .reset_fences(&[self.submit_fence])
                .expect("failed to reset cached command fence");
            let begin_info = vk::CommandBufferBeginInfo::default();
            self.inner
                .device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .expect("failed to begin cached command buffer");
        }

        let mut recorder = CommandRecorder {
            inner: self.inner.as_ref(),
            command_buffer: self.command_buffer,
            descriptor_sets: Vec::new(),
        };
        record(&mut recorder);

        unsafe {
            self.inner
                .device
                .end_command_buffer(self.command_buffer)
                .expect("failed to end cached command buffer");
        }

        if !self.descriptor_sets.is_empty() {
            let _pool_lock = self
                .inner
                .descriptor_pool_lock
                .lock()
                .expect("failed to lock descriptor pool for descriptor-set cleanup");
            unsafe {
                let _ = self
                    .inner
                    .device
                    .free_descriptor_sets(self.inner.descriptor_pool, &self.descriptor_sets);
            }
        }
        self.descriptor_sets = recorder.descriptor_sets;
    }

    pub(crate) fn submit_and_wait(&self) {
        let _submit_lock = self
            .inner
            .submit_lock
            .lock()
            .expect("failed to lock submit path");
        unsafe {
            self.inner
                .device
                .reset_fences(&[self.submit_fence])
                .expect("failed to reset cached command fence");
            let command_buffers = [self.command_buffer];
            let submit_info = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
            self.inner
                .device
                .queue_submit(self.inner.queue, &submit_info, self.submit_fence)
                .expect("failed to submit cached command buffer");
            self.inner
                .device
                .wait_for_fences(&[self.submit_fence], true, u64::MAX)
                .expect("failed to wait for cached command fence");
        }
    }
}

impl ComputeCommandContext {
    fn assert_same_device(&self, device: &ComputeDevice) {
        assert!(
            Arc::ptr_eq(&self.inner, &device.inner),
            "command context must be used with the compute device that created it"
        );
    }

    fn run_commands<F>(&mut self, record: F)
    where
        F: FnOnce(&mut CommandRecorder<'_>),
    {
        let _submit_lock = self
            .inner
            .submit_lock
            .lock()
            .expect("failed to lock submit path");

        unsafe {
            self.inner
                .device
                .reset_fences(&[self.submit_fence])
                .expect("failed to reset command context fence");
            self.inner
                .device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .expect("failed to reset command context buffer");
            let begin_info = vk::CommandBufferBeginInfo::default();
            self.inner
                .device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .expect("failed to begin command context buffer");
        }

        let mut recorder = CommandRecorder {
            inner: self.inner.as_ref(),
            command_buffer: self.command_buffer,
            descriptor_sets: Vec::new(),
        };
        record(&mut recorder);

        unsafe {
            self.inner
                .device
                .end_command_buffer(self.command_buffer)
                .expect("failed to end command context buffer");
        }

        let command_buffers = [self.command_buffer];
        let submit_info = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
        unsafe {
            self.inner
                .device
                .queue_submit(self.inner.queue, &submit_info, self.submit_fence)
                .expect("failed to submit command context buffer");
            self.inner
                .device
                .wait_for_fences(&[self.submit_fence], true, u64::MAX)
                .expect("failed to wait for command context fence");
        }

        if !recorder.descriptor_sets.is_empty() {
            let _pool_lock = self
                .inner
                .descriptor_pool_lock
                .lock()
                .expect("failed to lock descriptor pool");
            unsafe {
                let _ = self
                    .inner
                    .device
                    .free_descriptor_sets(self.inner.descriptor_pool, &recorder.descriptor_sets);
            }
        }
    }
}

impl ComputeDevice {
    pub fn new_default() -> Self {
        let entry = unsafe { ash::Entry::load().expect("failed to load Vulkan entry") };
        let app_name = CString::new("slangtag").expect("invalid app name");
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .engine_name(&app_name)
            .api_version(vk::API_VERSION_1_3);

        let available_layers = unsafe {
            entry
                .enumerate_instance_layer_properties()
                .expect("failed to enumerate Vulkan instance layers")
        };
        let validation_layer = CString::new("VK_LAYER_KHRONOS_validation").expect("invalid layer");
        let mut enabled_layers = Vec::new();
        if available_layers.iter().any(|layer| unsafe {
            CStr::from_ptr(layer.layer_name.as_ptr()) == validation_layer.as_c_str()
        }) {
            enabled_layers.push(validation_layer.as_ptr());
        }

        let instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&enabled_layers);

        let instance = unsafe {
            entry
                .create_instance(&instance_create_info, None)
                .expect("failed to create Vulkan instance")
        };

        let physical_device = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("failed to enumerate Vulkan physical devices")
                .into_iter()
                .next()
                .expect("no Vulkan physical devices available")
        };

        let queue_family_index = unsafe {
            instance
                .get_physical_device_queue_family_properties(physical_device)
                .iter()
                .position(|queue_family| queue_family.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .expect("failed to find a compute queue family") as u32
        };

        let mut subgroup_features = vk::PhysicalDeviceSubgroupSizeControlFeatures::default();
        let mut features2 =
            vk::PhysicalDeviceFeatures2::default().push_next(&mut subgroup_features);
        unsafe {
            instance.get_physical_device_features2(physical_device, &mut features2);
        }

        let mut subgroup_props = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        let mut properties2 =
            vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup_props);
        unsafe {
            instance.get_physical_device_properties2(physical_device, &mut properties2);
        }

        let _supported_features = features2.features;
        let requested_features = vk::PhysicalDeviceFeatures::default()
            .shader_int64(true)
            .robust_buffer_access(true);

        let min_storage_buffer_offset_alignment = properties2
            .properties
            .limits
            .min_storage_buffer_offset_alignment;
        let timestamp_period_ns = properties2.properties.limits.timestamp_period;
        let _ = properties2;
        let required_subgroup_size_stages = subgroup_props.required_subgroup_size_stages;
        let min_subgroup_size = subgroup_props.min_subgroup_size;
        let max_subgroup_size = subgroup_props.max_subgroup_size;
        let max_compute_workgroup_subgroups = subgroup_props.max_compute_workgroup_subgroups;
        assert!(
            min_subgroup_size <= 32 && max_subgroup_size >= 32,
            "Vulkan device must support subgroup size of at least 32"
        );

        let queue_priority = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priority)];

        assert!(
            subgroup_features.subgroup_size_control == vk::TRUE,
            "Vulkan device does not support subgroup size control"
        );

        let mut device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_features(&requested_features);

        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .shader_int8(true)
            .storage_buffer8_bit_access(true)
            .uniform_and_storage_buffer8_bit_access(true);

        device_create_info = device_create_info.push_next(&mut features12);

        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .synchronization2(true)
            .subgroup_size_control(true)
            .compute_full_subgroups(true);

        device_create_info = device_create_info.push_next(&mut features13);

        let device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .expect("failed to create Vulkan logical device")
        };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let descriptor_bindings: Vec<_> = (0..MAX_STORAGE_BINDINGS)
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let descriptor_set_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_bindings);
        let descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(&descriptor_set_layout_info, None)
                .expect("failed to create descriptor set layout")
        };

        let push_constant_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(MAX_PUSH_CONSTANT_BYTES)];
        let set_layouts = [descriptor_set_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_constant_ranges);
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .expect("failed to create compute pipeline layout")
        };

        let descriptor_pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(DESCRIPTOR_SET_CAPACITY * MAX_STORAGE_BINDINGS)];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .pool_sizes(&descriptor_pool_sizes)
            .max_sets(DESCRIPTOR_SET_CAPACITY);
        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .expect("failed to create descriptor pool")
        };

        let allocator_create_info =
            vk_mem::AllocatorCreateInfo::new(&instance, &device, physical_device);
        let allocator = unsafe {
            Allocator::new(allocator_create_info).expect("failed to create vk-mem allocator")
        };

        Self {
            inner: Arc::new(ComputeDeviceInner {
                entry,
                instance,
                device,
                queue,
                queue_family_index,
                descriptor_pool,
                descriptor_set_layout,
                pipeline_layout,
                allocator: ManuallyDrop::new(allocator),
                submit_lock: Mutex::new(()),
                descriptor_pool_lock: Mutex::new(()),
                subgroup_size_control_enabled: true,
                required_subgroup_size_stages,
                min_subgroup_size,
                max_subgroup_size,
                max_compute_workgroup_subgroups,
                min_storage_buffer_offset_alignment,
                timestamp_period_ns,
            }),
        }
    }

    pub fn queue_family_index(&self) -> u32 {
        self.inner.queue_family_index
    }

    pub fn create_command_context(&self) -> ComputeCommandContext {
        let (command_pool, command_buffer, submit_fence) = self.allocate_command_resources();
        ComputeCommandContext {
            inner: Arc::clone(&self.inner),
            command_pool,
            command_buffer,
            submit_fence,
        }
    }

    fn allocate_command_resources(&self) -> (vk::CommandPool, vk::CommandBuffer, vk::Fence) {
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.inner.queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe {
            self.inner
                .device
                .create_command_pool(&command_pool_info, None)
                .expect("failed to create command pool")
        };
        let command_buffer = unsafe {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            self.inner
                .device
                .allocate_command_buffers(&alloc_info)
                .expect("failed to allocate command buffer")[0]
        };
        let submit_fence = unsafe {
            self.inner
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .expect("failed to create command fence")
        };
        (command_pool, command_buffer, submit_fence)
    }

    pub(crate) fn create_cached_command_buffer(&self) -> CachedCommandBuffer {
        let (command_pool, command_buffer, submit_fence) = self.allocate_command_resources();

        CachedCommandBuffer {
            inner: Arc::clone(&self.inner),
            command_pool,
            command_buffer,
            submit_fence,
            descriptor_sets: Vec::new(),
        }
    }

    pub fn create_timestamp_query_pool(&self, query_count: u32) -> GpuQueryPool {
        assert!(query_count > 0, "query_count must be > 0");
        let create_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(query_count);
        let query_pool = unsafe {
            self.inner
                .device
                .create_query_pool(&create_info, None)
                .expect("failed to create timestamp query pool")
        };
        GpuQueryPool {
            device: Arc::clone(&self.inner),
            query_pool,
            query_type: vk::QueryType::TIMESTAMP,
            query_count,
        }
    }

    pub fn get_query_pool_results_u64(
        &self,
        query_pool: &GpuQueryPool,
        first_query: u32,
        query_count: u32,
    ) -> Vec<u64> {
        query_pool.assert_range(first_query, query_count);
        let mut results = vec![0u64; query_count as usize];
        unsafe {
            self.inner
                .device
                .get_query_pool_results(
                    query_pool.query_pool,
                    first_query,
                    &mut results,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
                .expect("failed to get query pool results");
        }
        results
    }

    pub fn min_storage_buffer_offset_alignment(&self) -> vk::DeviceSize {
        self.inner.min_storage_buffer_offset_alignment
    }

    pub fn timestamp_period_ns(&self) -> f32 {
        self.inner.timestamp_period_ns
    }

    pub fn supports_required_subgroup_size(&self, subgroup_size: u32, workgroup_size: u32) -> bool {
        self.inner.subgroup_size_control_enabled
            && self
                .inner
                .required_subgroup_size_stages
                .contains(vk::ShaderStageFlags::COMPUTE)
            && self.inner.min_subgroup_size <= subgroup_size
            && self.inner.max_subgroup_size >= subgroup_size
            && self
                .inner
                .max_compute_workgroup_subgroups
                .saturating_mul(subgroup_size)
                >= workgroup_size
    }

    pub fn create_compute_pipeline(
        &self,
        module_words: &[u32],
        required_subgroup_size: Option<u32>,
    ) -> ComputePipeline {
        let shader_info = vk::ShaderModuleCreateInfo::default().code(module_words);
        let shader_module = unsafe {
            self.inner
                .device
                .create_shader_module(&shader_info, None)
                .expect("failed to create shader module")
        };

        let mut subgroup_info = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default();
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(main_entry_name());
        if let Some(size) = required_subgroup_size {
            assert!(
                self.supports_required_subgroup_size(size, size),
                "required subgroup size {size} is not supported on this Vulkan device"
            );
            subgroup_info = subgroup_info.required_subgroup_size(size);
            stage = stage.push_next(&mut subgroup_info);
        }

        let pipeline_info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(self.inner.pipeline_layout)];
        let pipeline = unsafe {
            self.inner
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
                .expect("failed to create compute pipeline")[0]
        };

        unsafe {
            self.inner.device.destroy_shader_module(shader_module, None);
        }

        ComputePipeline {
            device: Arc::clone(&self.inner),
            pipeline,
        }
    }

    pub fn create_buffer<T: Pod + Copy>(
        &self,
        len: usize,
        usage: vk::BufferUsageFlags,
        memory: BufferMemory,
    ) -> GpuBuffer<T> {
        let logical_len = len;
        let allocated_len = logical_len.max(1);
        let bytes = (allocated_len * std::mem::size_of::<T>()) as vk::DeviceSize;
        let create_info = vk::BufferCreateInfo::default().size(bytes).usage(usage);

        let mut alloc_info = AllocationCreateInfo::default();
        match memory {
            BufferMemory::DeviceLocal => {
                alloc_info.usage = MemoryUsage::AutoPreferDevice;
            }
            BufferMemory::HostSequentialWrite => {
                alloc_info.usage = MemoryUsage::AutoPreferHost;
                alloc_info.flags = AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE;
            }
            BufferMemory::HostRandomAccess => {
                alloc_info.usage = MemoryUsage::AutoPreferHost;
                alloc_info.flags = AllocationCreateFlags::HOST_ACCESS_RANDOM;
            }
        }

        let (buffer, allocation) = unsafe {
            self.inner
                .allocator
                .create_buffer(&create_info, &alloc_info)
                .expect("failed to allocate Vulkan buffer")
        };

        GpuBuffer {
            raw: Arc::new(RawBuffer {
                device: Arc::clone(&self.inner),
                buffer,
                allocation: Mutex::new(allocation),
                bytes,
            }),
            len: logical_len,
            _marker: PhantomData,
        }
    }

    pub fn upload_buffer<T: Pod + Copy>(
        &self,
        command_context: &mut ComputeCommandContext,
        data: &[T],
        usage: vk::BufferUsageFlags,
        prefer_device: bool,
    ) -> GpuBuffer<T> {
        command_context.assert_same_device(self);
        if prefer_device {
            let gpu_buffer = self.create_buffer::<T>(
                data.len(),
                usage | vk::BufferUsageFlags::TRANSFER_DST,
                BufferMemory::DeviceLocal,
            );
            let staging = self.create_buffer::<T>(
                data.len(),
                vk::BufferUsageFlags::TRANSFER_SRC,
                BufferMemory::HostSequentialWrite,
            );
            staging.write(data);
            self.copy_buffer(
                command_context,
                &staging,
                &gpu_buffer,
                (data.len() * std::mem::size_of::<T>()) as vk::DeviceSize,
            );
            gpu_buffer
        } else {
            let host_buffer =
                self.create_buffer::<T>(data.len(), usage, BufferMemory::HostSequentialWrite);
            host_buffer.write(data);
            host_buffer
        }
    }

    pub fn copy_buffer<T: Pod + Copy>(
        &self,
        command_context: &mut ComputeCommandContext,
        src: &GpuBuffer<T>,
        dst: &GpuBuffer<T>,
        bytes: vk::DeviceSize,
    ) {
        self.copy_buffer_region(command_context, src, 0, dst, 0, bytes);
    }

    pub fn copy_buffer_region<T: Pod + Copy>(
        &self,
        command_context: &mut ComputeCommandContext,
        src: &GpuBuffer<T>,
        src_offset: vk::DeviceSize,
        dst: &GpuBuffer<T>,
        dst_offset: vk::DeviceSize,
        bytes: vk::DeviceSize,
    ) {
        command_context.assert_same_device(self);
        command_context.run_commands(|recorder| {
            recorder.copy_buffer_region(src, src_offset, dst, dst_offset, bytes);
        });
    }

    pub fn copy_descriptor_buffer_to_buffer<T: Pod + Copy>(
        &self,
        command_context: &mut ComputeCommandContext,
        src: DescriptorBuffer,
        dst: &GpuBuffer<T>,
        bytes: vk::DeviceSize,
    ) {
        assert!(
            bytes <= src.range,
            "copy source range exceeds descriptor range: bytes={} src_range={}",
            bytes,
            src.range
        );
        assert!(
            bytes <= dst.byte_size(),
            "copy destination range exceeds buffer size: bytes={} dst_size={}",
            bytes,
            dst.byte_size()
        );
        command_context.assert_same_device(self);
        command_context.run_commands(|recorder| unsafe {
            let copy_regions = [vk::BufferCopy::default()
                .src_offset(src.offset)
                .dst_offset(0)
                .size(bytes)];
            recorder.inner.device.cmd_copy_buffer(
                recorder.command_buffer,
                src.buffer,
                dst.raw.buffer,
                &copy_regions,
            );
        });
    }

    pub fn fill_buffer_u32(
        &self,
        command_context: &mut ComputeCommandContext,
        buffer: &GpuBuffer<u32>,
        value: u32,
    ) {
        self.fill_buffer_u32_range(command_context, buffer, 0, buffer.raw.bytes, value);
    }

    pub fn fill_buffer_u32_range(
        &self,
        command_context: &mut ComputeCommandContext,
        buffer: &GpuBuffer<u32>,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
        value: u32,
    ) {
        command_context.assert_same_device(self);
        command_context.run_commands(|recorder| {
            recorder.fill_buffer_u32_range(buffer, offset, size, value);
        });
    }

    pub fn dispatch_with_push_constants<T: Pod + Copy>(
        &self,
        command_context: &mut ComputeCommandContext,
        pipeline: &ComputePipeline,
        bindings: &[(u32, DescriptorBuffer)],
        push_constants: &T,
        groups: [u32; 3],
    ) {
        command_context.assert_same_device(self);
        command_context.run_commands(|recorder| {
            recorder.dispatch_with_push_constants(pipeline, bindings, push_constants, groups);
        });
    }

    pub(crate) fn run_commands<F>(&self, command_context: &mut ComputeCommandContext, record: F)
    where
        F: FnOnce(&mut CommandRecorder<'_>),
    {
        command_context.assert_same_device(self);
        command_context.run_commands(record);
    }
}

impl<T: Pod + Copy> GpuBuffer<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn write(&self, data: &[T]) {
        let byte_len = (data.len() * std::mem::size_of::<T>()) as vk::DeviceSize;
        assert!(
            byte_len <= self.raw.bytes,
            "write exceeds allocation: {} > {} bytes",
            byte_len,
            self.raw.bytes
        );

        let mut allocation = self
            .raw
            .allocation
            .lock()
            .expect("failed to lock allocation for write");
        let mapped_ptr = unsafe {
            self.raw
                .device
                .allocator
                .map_memory(&mut allocation)
                .expect("failed to map allocation for write")
        };
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr() as *const u8, mapped_ptr, byte_len as usize);
            self.raw
                .device
                .allocator
                .flush_allocation(&allocation, 0, byte_len)
                .expect("failed to flush allocation after write");
            self.raw.device.allocator.unmap_memory(&mut allocation);
        }
    }

    pub fn with_mapped_bytes_mut<E, F, R>(&self, f: F) -> Result<R, E>
    where
        F: FnOnce(*mut u8, usize) -> Result<R, E>,
    {
        let byte_len = self.raw.bytes;
        let mut allocation = self
            .raw
            .allocation
            .lock()
            .expect("failed to lock allocation for mapped write");
        let mapped_ptr = unsafe {
            self.raw
                .device
                .allocator
                .map_memory(&mut allocation)
                .expect("failed to map allocation for mapped write")
        };
        let result = f(mapped_ptr, byte_len as usize);
        unsafe {
            self.raw
                .device
                .allocator
                .flush_allocation(&allocation, 0, byte_len)
                .expect("failed to flush allocation after mapped write");
            self.raw.device.allocator.unmap_memory(&mut allocation);
        }
        result
    }

    pub fn read(&self, len: usize) -> Vec<T> {
        let read_len = len.min(self.len);
        if read_len == 0 {
            return Vec::new();
        }

        let byte_len = (read_len * std::mem::size_of::<T>()) as vk::DeviceSize;
        assert!(
            byte_len <= self.raw.bytes,
            "read exceeds allocation: {} > {} bytes",
            byte_len,
            self.raw.bytes
        );

        let mut allocation = self
            .raw
            .allocation
            .lock()
            .expect("failed to lock allocation for read");
        let mapped_ptr = unsafe {
            self.raw
                .device
                .allocator
                .map_memory(&mut allocation)
                .expect("failed to map allocation for read")
        };
        unsafe {
            self.raw
                .device
                .allocator
                .invalidate_allocation(&allocation, 0, byte_len)
                .expect("failed to invalidate allocation before read");
            let typed_ptr = mapped_ptr as *const T;
            let slice = std::slice::from_raw_parts(typed_ptr, read_len);
            let out = slice.to_vec();
            self.raw.device.allocator.unmap_memory(&mut allocation);
            out
        }
    }

    pub fn descriptor(&self) -> DescriptorBuffer {
        DescriptorBuffer::from(self)
    }

    pub fn descriptor_range(
        &self,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) -> DescriptorBuffer {
        assert!(
            offset.saturating_add(range) <= self.raw.bytes,
            "descriptor range exceeds allocation: offset={} range={} size={}",
            offset,
            range,
            self.raw.bytes
        );
        DescriptorBuffer {
            buffer: self.raw.buffer,
            offset,
            range,
        }
    }

    pub fn byte_size(&self) -> vk::DeviceSize {
        self.raw.bytes
    }
}

impl GpuQueryPool {
    pub fn query_count(&self) -> u32 {
        self.query_count
    }

    fn assert_range(&self, first_query: u32, query_count: u32) {
        let end = first_query.checked_add(query_count).unwrap_or_else(|| {
            panic!("query range overflow: first={first_query}, count={query_count}")
        });
        assert!(
            end <= self.query_count,
            "query range out of bounds: first={} count={} total={}",
            first_query,
            query_count,
            self.query_count
        );
    }
}
