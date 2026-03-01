use crate::{ComputeDevice, GPUImage, Size, compute_shader_path};
use bytemuck::{Pod, Zeroable};
use image::{DynamicImage, GrayImage, ImageBuffer};
use num_traits::ToPrimitive;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::sync::{self, GpuFuture};
use vulkano::{
    buffer::{Buffer, Subbuffer},
    pipeline::{
        ComputePipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    shader::{ShaderModuleCreateInfo, spirv::bytes_to_words},
};

pub struct DetectionSettings {
    pub decimate: Option<u8>,
    pub min_white_black_diff: u8,
}

pub struct Detector {
    device: ComputeDevice,
    settings: DetectionSettings,
}

impl Default for DetectionSettings {
    fn default() -> Self {
        Self {
            decimate: Some(2),
            min_white_black_diff: 15,
        }
    }
}

fn is_power_of_two(n: u8) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct DecimatePushConstants {
    input_size: Size,
    decimated_size: Size,
}

impl Detector {
    pub fn new(device: ComputeDevice, settings: DetectionSettings) -> Self {
        Self { device, settings }
    }

    pub fn detect(&self, image: DynamicImage) -> Result<(), ()> {
        let gray = image.into_luma8();
        let raw_gpu_image = crate::GPUImage::from_image_buffer(self.device.clone(), gray);

        let gpu_image = match self.settings.decimate {
            Some(factor) => {
                if !is_power_of_two(factor) {
                    return Err(()); // Invalid decimation factor
                }
                let new_img = self.decimate(&raw_gpu_image, factor);
                drop(raw_gpu_image); // Explicitly drop the original GPU image to free resources
                new_img
            }
            None => raw_gpu_image,
        };

        

        Ok(())
    }

    fn decimate(&self, image: &GPUImage<u8>, factor: u8) -> GPUImage<u8> {
        let module_bytes = include_bytes!(compute_shader_path!("01-threshold-decimate"));

        let shader = unsafe {
            vulkano::shader::ShaderModule::new(
                self.device.device.clone(),
                ShaderModuleCreateInfo::new(&bytes_to_words(module_bytes).unwrap()),
            )
            .expect("failed to create shader module")
        };

        let entry_point = shader
            .entry_point("main")
            .expect("failed to find entry point in shader");

        let stage = PipelineShaderStageCreateInfo::new(entry_point);
        let layout = PipelineLayout::new(
            self.device.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(self.device.device.clone())
                .unwrap(),
        )
        .unwrap();

        let compute_pipeline = ComputePipeline::new(
            self.device.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .expect("failed to create compute pipeline");

        let decimated_size = crate::Size::new(
            image.size.width / factor as u32,
            image.size.height / factor as u32,
        );
        let decimated_image_buffer: Subbuffer<[u8]> = Buffer::new_unsized(
            self.device.memory_allocator.clone(),
            vulkano::buffer::BufferCreateInfo {
                usage: vulkano::buffer::BufferUsage::STORAGE_BUFFER
                    | vulkano::buffer::BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            vulkano::memory::allocator::AllocationCreateInfo {
                memory_type_filter: vulkano::memory::allocator::MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            (decimated_size.total_pixels() as usize).try_into().unwrap(),
        )
        .unwrap();

        let pipeline_layout = compute_pipeline.layout();
        let descriptor_set_layouts = pipeline_layout.set_layouts();

        let descriptor_set_layout_index = 0;
        let descriptor_set_layout = descriptor_set_layouts
            .get(descriptor_set_layout_index)
            .unwrap();
        let descriptor_set = DescriptorSet::new(
            self.device.descriptor_set_allocator.clone(),
            descriptor_set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, image.image.clone()),
                WriteDescriptorSet::buffer(1, decimated_image_buffer.clone()),
            ],
            [],
        )
        .unwrap();

        let mut builder = vulkano::command_buffer::AutoCommandBufferBuilder::primary(
            self.device.command_buffer_allocator.clone(),
            self.device.queue_family_index,
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let push_constants = DecimatePushConstants {
            input_size: image.size,
            decimated_size,
        };

        unsafe {
            builder
                .bind_pipeline_compute(compute_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    vulkano::pipeline::PipelineBindPoint::Compute,
                    compute_pipeline.layout().clone(),
                    descriptor_set_layout_index as u32,
                    descriptor_set.clone(),
                )
                .unwrap()
                .push_constants(compute_pipeline.layout().clone(), 0, push_constants)
                .unwrap()
                .dispatch([
                    image.size.width,
                    image.size.height,
                    1,
                ])
                .unwrap()
        };

        let command_buffer = builder.build().unwrap();

        let future = sync::now(self.device.device.clone())
            .then_execute(self.device.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();

        GPUImage::new(self.device.clone(), decimated_image_buffer, decimated_size)
    }
}
