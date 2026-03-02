pub mod detect;

#[macro_export]
macro_rules! compute_shader_path {
    ($shader_stem:literal) => {
        concat!(env!("OUT_DIR"), "/shaders/compute/", $shader_stem, ".spv")
    };
}

use bytemuck::{Pod, Zeroable};
use image::{DynamicImage, ImageBuffer, Pixel, Primitive};
use num_traits::real::Real;
use std::fmt::Debug;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::{fs, io, process::Command};
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo};
use vulkano::device::{DeviceFeatures, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::sync::GpuFuture;
use vulkano::{VulkanLibrary, sync};

#[repr(C)]
#[derive(Pod, Zeroable, Clone, Copy)]
pub struct Size {
    pub width: u32,
    pub height: u32,
}

impl Size {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub fn total_pixels(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }
}

/// GPUImage is a grayscale image in GPU memory.
/// T is a marker type for the image format, e.g. u8 for 8-bit grayscale, f32 for 32-bit float grayscale, etc.
#[derive(Clone)]
pub struct GPUImage<T> {
    pub(crate) image: Subbuffer<[T]>,
    pub(crate) size: Size,
    device: ComputeDevice,

    _marker: std::marker::PhantomData<T>,
}

impl<T: BufferContents + Primitive + Debug> GPUImage<T> {
    pub fn new(device: ComputeDevice, image: Subbuffer<[T]>, size: Size) -> Self {
        Self {
            image,
            size,
            device,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn from_image_buffer(
        device: ComputeDevice,
        image: ImageBuffer<image::Luma<T>, Vec<T>>,
    ) -> Self {
        let size = Size::new(image.width(), image.height());
        let flat_data: Vec<T> = image.into_raw();

        // Create a staging buffer for the image data
        let transfer_buffer: Subbuffer<[T]> = Buffer::from_iter(
            device.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            flat_data.into_iter(),
        )
        .expect("failed to create transfer buffer");

        // Create the GPU image buffer that will hold the data on the GPU
        // This buffer should be device-local for optimal performance, and we'll copy the data from the staging buffer
        let gpu_image: Subbuffer<[T]> = Buffer::new_unsized(
            device.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            size.total_pixels()
                .try_into()
                .expect("image size exceeds buffer limits"),
        )
        .expect("failed to create GPU image buffer");

        // Copy data from the staging buffer to the GPU image buffer
        let mut builder = AutoCommandBufferBuilder::primary(
            device.command_buffer_allocator.clone(),
            device.queue_family_index,
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                transfer_buffer.clone(),
                gpu_image.clone(),
            ))
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let future = sync::now(device.device.clone())
            .then_execute(device.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();

        Self {
            image: gpu_image,
            size,
            device,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn data(&self) -> Vec<T> {
        let destination: Subbuffer<[T]> = Buffer::new_unsized(
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
            self.size
                .total_pixels()
                .try_into()
                .expect("image size exceeds buffer limits"),
        )
        .expect("failed to create destination buffer");

        let mut builder = AutoCommandBufferBuilder::primary(
            self.device.command_buffer_allocator.clone(),
            self.device.queue_family_index,
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                self.image.clone(),
                destination.clone(),
            ))
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let future = sync::now(self.device.device.clone())
            .then_execute(self.device.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush() // same as signal fence, and then flush
            .unwrap();

        future.wait(None).unwrap();

        let data = destination.read().unwrap();

        data.to_vec()
    }

    pub fn to_image_buffer(&self) -> ImageBuffer<image::Luma<T>, Vec<T>> {
        let data = self.data();
        println!("Read {} pixels from GPU image", data.len());
        println!("Pixel data sample: {:?}", data.get(0..10));
        ImageBuffer::from_raw(self.size.width, self.size.height, data)
            .expect("failed to create image buffer from GPU image data")
    }
}

#[derive(Clone)]
pub struct ComputeDevice {
    pub(crate) device: Arc<vulkano::device::Device>,
    pub(crate) queue: Arc<vulkano::device::Queue>,
    pub(crate) memory_allocator: Arc<vulkano::memory::allocator::StandardMemoryAllocator>,
    pub(crate) command_buffer_allocator:
        Arc<vulkano::command_buffer::allocator::StandardCommandBufferAllocator>,
    pub(crate) descriptor_set_allocator:
        Arc<vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator>,
    pub(crate) queue_family_index: u32,
}

impl ComputeDevice {
    pub fn new_default() -> ComputeDevice {
        let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..Default::default()
            },
        )
        .expect("failed to create instance");

        let physical_device = instance
            .enumerate_physical_devices()
            .expect("could not enumerate devices")
            .next()
            .expect("no devices available");

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .position(|queue_family_properties| {
                queue_family_properties
                    .queue_flags
                    .contains(QueueFlags::COMPUTE)
            })
            .expect("couldn't find a graphical queue family")
            as u32;

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                // here we pass the desired queue family to use by index
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_features: DeviceFeatures {
                    shader_int8: true,  // >90% support.
                    shader_int64: true, // 56% support. mostly android
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .expect("failed to create device");
        let queue = queues.next().unwrap();

        let memory_allocator = Arc::new(
            vulkano::memory::allocator::StandardMemoryAllocator::new_default(device.clone()),
        );
        let command_buffer_allocator = Arc::new(
            vulkano::command_buffer::allocator::StandardCommandBufferAllocator::new(
                device.clone(),
                vulkano::command_buffer::allocator::StandardCommandBufferAllocatorCreateInfo::default(),
            ),
        );
        let descriptor_set_allocator = Arc::new(
            vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator::new(
                device.clone(),
                Default::default(),
            ),
        );

        ComputeDevice {
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
            queue_family_index,
        }
    }
}
