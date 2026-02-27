pub mod detect;

use std::sync::Arc;
use vulkano::VulkanLibrary;
use vulkano::device::QueueFlags;
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};

pub struct ComputeDevice {
    pub(crate) device: Arc<vulkano::device::Device>,
    pub(crate) queue: Arc<vulkano::device::Queue>,
    pub(crate) memory_allocator: Arc<vulkano::memory::allocator::StandardMemoryAllocator>,
    pub(crate) command_buffer_allocator:
        Arc<vulkano::command_buffer::allocator::StandardCommandBufferAllocator>,
    pub(crate) descriptor_set_allocator:
        Arc<vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator>,
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
                ..Default::default()
            },
        )
        .expect("failed to create device");
        let queue = queues.next().unwrap();

        let memory_allocator = Arc::new(vulkano::memory::allocator::StandardMemoryAllocator::new_default(
            device.clone(),
        ));
        let command_buffer_allocator = Arc::new(
            vulkano::command_buffer::allocator::StandardCommandBufferAllocator::new(
                device.clone(),
                vulkano::command_buffer::allocator::StandardCommandBufferAllocatorCreateInfo::default(),
            ),
        );
        let descriptor_set_allocator = Arc::new(
            vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator::new(device.clone(), Default::default()),
        );

        ComputeDevice {
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
        }
    }
}
