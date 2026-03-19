pub mod detect;
pub mod gpu;
pub mod sort;

#[macro_export]
macro_rules! compute_shader_path {
    ($shader_stem:literal) => {
        concat!(env!("OUT_DIR"), "/shaders/compute/", $shader_stem, ".spv")
    };
}

#[macro_export]
macro_rules! include_u32 {
    ($path:expr $(,)?) => {{
        // 1. Get a slice of the bytes to check its length
        const BYTES: &[u8] = include_bytes!($path);
        const LEN: usize = BYTES.len();

        // 2. Compile-time assertion: ensure length is divisible by 4
        // If the file is the wrong size, the compiler will hard-stop here.
        const _: () = assert!(
            LEN % 4 == 0,
            "Included file length must be a multiple of 4 bytes"
        );

        // 3. Define a wrapper struct that forces 4-byte alignment
        #[repr(C, align(4))]
        struct Aligned<const N: usize>([u8; N]);

        // 4. Bake the included data into the binary inside our aligned struct.
        // `include_bytes!` returns `&[u8; N]`, so we dereference it to store the array.
        static ALIGNED_DATA: Aligned<LEN> = Aligned(*include_bytes!($path));

        // 5. Safely cast the aligned byte pointer to a u32 slice
        unsafe { std::slice::from_raw_parts(ALIGNED_DATA.0.as_ptr().cast::<u32>(), LEN / 4) }
    }};
}

use bytemuck::{Pod, Zeroable};
use gpu::{BufferMemory, GpuBuffer};
use std::fmt::Debug;

pub use gpu::{ComputeDevice, ComputePipeline, DescriptorBuffer, GpuQueryPool};

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

#[derive(Clone)]
pub struct GPUImage<T: Pod + Copy> {
    pub(crate) image: GpuBuffer<T>,
    pub(crate) size: Size,
    device: ComputeDevice,
}

impl<T: Pod + Copy + Debug> GPUImage<T> {
    pub fn new(device: ComputeDevice, image: GpuBuffer<T>, size: Size) -> Self {
        Self {
            image,
            size,
            device,
        }
    }

    pub fn from_vec(device: ComputeDevice, size: Size, data: Vec<T>) -> Self {
        let pixel_count = size.total_pixels();
        assert_eq!(
            data.len(),
            pixel_count,
            "input data length ({}) must match image size ({})",
            data.len(),
            pixel_count
        );

        let transfer = device.upload_buffer(&data, ash::vk::BufferUsageFlags::TRANSFER_SRC, false);
        let gpu_image = device.create_buffer::<T>(
            pixel_count,
            ash::vk::BufferUsageFlags::STORAGE_BUFFER
                | ash::vk::BufferUsageFlags::TRANSFER_DST
                | ash::vk::BufferUsageFlags::TRANSFER_SRC,
            BufferMemory::DeviceLocal,
        );
        device.copy_buffer(
            &transfer,
            &gpu_image,
            (pixel_count * std::mem::size_of::<T>()) as ash::vk::DeviceSize,
        );

        Self {
            image: gpu_image,
            size,
            device,
        }
    }

    pub fn from_vec_fast(device: ComputeDevice, size: Size, data: Vec<T>) -> Self {
        let pixel_count = size.total_pixels();
        assert_eq!(
            data.len(),
            pixel_count,
            "input data length ({}) must match image size ({})",
            data.len(),
            pixel_count
        );

        let gpu_image = device.upload_buffer(
            &data,
            ash::vk::BufferUsageFlags::STORAGE_BUFFER | ash::vk::BufferUsageFlags::TRANSFER_SRC,
            true,
        );

        Self {
            image: gpu_image,
            size,
            device,
        }
    }

    pub fn data(&self) -> Vec<T> {
        let pixel_count = self.size.total_pixels();
        let destination = self.device.create_buffer::<T>(
            pixel_count,
            ash::vk::BufferUsageFlags::TRANSFER_DST,
            BufferMemory::HostRandomAccess,
        );
        self.device.copy_buffer(
            &self.image,
            &destination,
            (pixel_count * std::mem::size_of::<T>()) as ash::vk::DeviceSize,
        );
        destination.read(pixel_count)
    }
}
