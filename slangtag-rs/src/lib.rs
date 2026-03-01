pub mod detect;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::{fs, io, process::Command};
use vulkano::VulkanLibrary;
use vulkano::device::QueueFlags;
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};

/// Compiles a Slang compute shader to SPIR-V and returns the output path.
///
/// This uses `slangc` from `$SLANGC` if set, otherwise `slangc` from `PATH`.
/// Relative `shader_path` values are resolved from `$CARGO_MANIFEST_DIR` when
/// available.
pub fn compile_slang_compute_shader_path<P: AsRef<Path>>(
    shader_path: P,
    entry_point: &str,
) -> io::Result<PathBuf> {
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR").map(PathBuf::from);
    let raw_shader_path = shader_path.as_ref();
    let shader_path = if raw_shader_path.is_absolute() {
        raw_shader_path.to_path_buf()
    } else if let Some(manifest_dir) = manifest_dir {
        manifest_dir.join(raw_shader_path)
    } else {
        raw_shader_path.to_path_buf()
    };

    let shader_stem = shader_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid shader filename"))?;

    let out_root = std::env::var_os("OUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join("slangtag-spv");
    fs::create_dir_all(&out_root)?;
    let out_path = out_root.join(format!("{shader_stem}.{entry_point}.spv"));

    let compiler = std::env::var("SLANGC").unwrap_or_else(|_| "slangc".to_string());
    let output = Command::new(compiler)
        .arg(&shader_path)
        .arg("-target")
        .arg("spirv")
        .arg("-stage")
        .arg("compute")
        .arg("-entry")
        .arg(entry_point)
        .arg("-o")
        .arg(&out_path)
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(io::Error::other(format!(
            "slangc failed for {} (entry: {}): {}",
            shader_path.display(),
            entry_point,
            stderr.trim()
        )));
    }

    Ok(out_path)
}

#[macro_export]
macro_rules! compile_slang_compute_shader {
    ($shader_path:expr) => {
        $crate::compile_slang_compute_shader_path($shader_path, "main")
    };
    ($shader_path:expr, $entry_point:expr) => {
        $crate::compile_slang_compute_shader_path($shader_path, $entry_point)
    };
}

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
