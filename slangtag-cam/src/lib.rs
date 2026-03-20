use ash::vk;
use slangtag::detect::{DetectError, Detector};
use slangtag::gpu::{BufferMemory, GpuBuffer};
use slangtag::{ComputeDevice, Size};
use std::ffi::CStr;
use std::os::raw::c_ulong;
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use thiserror::Error;
use turbojpeg_sys::{
    TJFLAG_FASTDCT, TJFLAG_FASTUPSAMPLE, TJPF_TJPF_GRAY, tjDecompress2, tjDecompressHeader3,
    tjDestroy, tjGetErrorStr2, tjInitDecompress, tjhandle,
};
use v4l2r::device::queue::direction::Capture;
use v4l2r::device::queue::{GetFreeCaptureBuffer, Queue};
use v4l2r::device::{AllocatedQueue, Device, DeviceConfig, Stream, TryDequeue};
use v4l2r::ioctl::{DqBufIoctlError, IoctlConvertError};
use v4l2r::memory::MmapHandle;

pub use slangtag::detect::{DetectedTag, DetectionSettings};

const CAMERA_BUFFER_COUNT: u32 = 4;
const PIPELINE_SLOTS: usize = 3;

#[derive(Debug, Clone)]
pub struct CameraConfig {
    pub device_path: String,
    pub width: u32,
    pub height: u32,
    pub fps: Option<u32>,
}

impl CameraConfig {
    pub fn new(device_path: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            device_path: device_path.into(),
            width,
            height,
            fps: None,
        }
    }
}

#[derive(Debug, Error, Clone)]
pub enum CamError {
    #[error("camera width/height must be non-zero")]
    InvalidCameraSize,
    #[error("camera format not accepted: expected MJPG/JPEG, got {0}")]
    UnsupportedPixelFormat(String),
    #[error(
        "camera format size mismatch: requested {requested_width}x{requested_height}, got {actual_width}x{actual_height}"
    )]
    CameraSizeMismatch {
        requested_width: u32,
        requested_height: u32,
        actual_width: u32,
        actual_height: u32,
    },
    #[error("camera size is invalid after detector alignment")]
    InvalidAlignedSize,
    #[error("v4l2 setup failed: {0}")]
    V4l2Setup(String),
    #[error("v4l2 runtime failure: {0}")]
    V4l2Runtime(String),
    #[error("v4l2 stream ended")]
    EndOfStream,
    #[error("failed to map v4l2 capture buffer")]
    CaptureMappingFailed,
    #[error("jpeg decode failed: {0}")]
    JpegDecode(String),
    #[error(
        "jpeg size does not match configured camera size: expected {expected_width}x{expected_height}, got {actual_width}x{actual_height}"
    )]
    JpegSizeMismatch {
        expected_width: u32,
        expected_height: u32,
        actual_width: u32,
        actual_height: u32,
    },
    #[error("detector initialization failed: {0}")]
    DetectorInit(String),
    #[error("detector execution failed: {0}")]
    Detect(String),
    #[error("capture worker stopped")]
    WorkerStopped,
}

#[derive(Debug, Clone, Copy)]
struct ReadyFrame {
    slot: usize,
    frame_id: u64,
    timestamp_ns: u64,
}

#[derive(Debug)]
struct SharedState {
    free_slots: Vec<usize>,
    latest_ready: Option<ReadyFrame>,
    fatal_error: Option<CamError>,
    shutdown: bool,
}

impl SharedState {
    fn new() -> Self {
        let mut free_slots = Vec::with_capacity(PIPELINE_SLOTS);
        for i in 0..PIPELINE_SLOTS {
            free_slots.push(i);
        }
        Self {
            free_slots,
            latest_ready: None,
            fatal_error: None,
            shutdown: false,
        }
    }
}

struct SharedQueue {
    state: Mutex<SharedState>,
    cv: Condvar,
}

impl SharedQueue {
    fn new() -> Self {
        Self {
            state: Mutex::new(SharedState::new()),
            cv: Condvar::new(),
        }
    }
}

#[derive(Clone)]
struct FrameSlot {
    buffer: GpuBuffer<u8>,
    size: Size,
}

impl FrameSlot {
    fn new(device: &ComputeDevice, size: Size) -> Self {
        let len = size.total_pixels();
        let buffer = device.create_buffer(
            len,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            BufferMemory::HostSequentialWrite,
        );
        Self { buffer, size }
    }
}

pub struct CameraTagStream {
    detector: Detector,
    slots: Arc<Vec<FrameSlot>>,
    shared: Arc<SharedQueue>,
    worker: Option<JoinHandle<()>>,
    terminated: bool,
}

impl CameraTagStream {
    pub fn new(
        device: ComputeDevice,
        config: CameraConfig,
        detection: DetectionSettings,
    ) -> Result<Self, CamError> {
        if config.width == 0 || config.height == 0 {
            return Err(CamError::InvalidCameraSize);
        }

        let detector_size =
            normalize_size(Size::new(config.width, config.height), detection.decimate)
                .ok_or(CamError::InvalidAlignedSize)?;
        let detector = Detector::new(device.clone(), detection, detector_size)
            .map_err(|err| CamError::DetectorInit(err.to_string()))?;

        let mut slots = Vec::with_capacity(PIPELINE_SLOTS);
        for _ in 0..PIPELINE_SLOTS {
            slots.push(FrameSlot::new(&device, detector_size));
        }
        let slots = Arc::new(slots);
        let shared = Arc::new(SharedQueue::new());

        let worker_shared = Arc::clone(&shared);
        let worker_slots = Arc::clone(&slots);
        let worker_config = config.clone();
        let worker = thread::Builder::new()
            .name("slangtag-cam-capture".to_string())
            .spawn(move || {
                if let Err(err) = run_capture_decode_worker(
                    worker_config,
                    worker_slots,
                    Arc::clone(&worker_shared),
                ) {
                    if let Ok(mut guard) = worker_shared.state.lock() {
                        if !guard.shutdown {
                            guard.fatal_error = Some(err);
                        }
                        worker_shared.cv.notify_all();
                    }
                }
            })
            .map_err(|err| CamError::V4l2Setup(err.to_string()))?;

        Ok(Self {
            detector,
            slots,
            shared,
            worker: Some(worker),
            terminated: false,
        })
    }
}

impl Iterator for CameraTagStream {
    type Item = Result<Vec<DetectedTag>, CamError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.terminated {
            return None;
        }

        let ready = {
            let mut guard = match self.shared.state.lock() {
                Ok(guard) => guard,
                Err(_) => {
                    self.terminated = true;
                    return Some(Err(CamError::WorkerStopped));
                }
            };
            loop {
                if let Some(err) = guard.fatal_error.take() {
                    self.terminated = true;
                    return Some(Err(err));
                }
                if let Some(ready) = guard.latest_ready.take() {
                    break ready;
                }
                if guard.shutdown {
                    self.terminated = true;
                    return None;
                }
                guard = match self.shared.cv.wait(guard) {
                    Ok(guard) => guard,
                    Err(_) => {
                        self.terminated = true;
                        return Some(Err(CamError::WorkerStopped));
                    }
                };
            }
        };

        let slot = &self.slots[ready.slot];
        let detect_result = self
            .detector
            .detect_descriptor(slot.buffer.descriptor(), slot.size)
            .map(|output| output.tags)
            .map_err(map_detect_error);

        if let Ok(mut guard) = self.shared.state.lock() {
            guard.free_slots.push(ready.slot);
            self.shared.cv.notify_all();
        } else {
            self.terminated = true;
            return Some(Err(CamError::WorkerStopped));
        }

        Some(detect_result)
    }
}

impl Drop for CameraTagStream {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.shared.state.lock() {
            guard.shutdown = true;
            self.shared.cv.notify_all();
        }
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }
}

fn normalize_size(size: Size, decimate: Option<u8>) -> Option<Size> {
    let decimate = decimate.unwrap_or(1) as u32;
    let multiple = 4 * decimate;
    let width = size.width - (size.width % multiple);
    let height = size.height - (size.height % multiple);
    if width == 0 || height == 0 {
        None
    } else {
        Some(Size::new(width, height))
    }
}

fn map_detect_error(err: DetectError) -> CamError {
    CamError::Detect(err.to_string())
}

fn run_capture_decode_worker(
    config: CameraConfig,
    slots: Arc<Vec<FrameSlot>>,
    shared: Arc<SharedQueue>,
) -> Result<(), CamError> {
    let device = Device::open(
        Path::new(&config.device_path),
        DeviceConfig::new().non_blocking_dqbuf(),
    )
    .map_err(|err| CamError::V4l2Setup(err.to_string()))?;
    let device = Arc::new(device);

    let mut capture_queue =
        if let Ok(q) = Queue::<Capture, _>::get_capture_queue(Arc::clone(&device)) {
            q
        } else {
            Queue::<Capture, _>::get_capture_mplane_queue(Arc::clone(&device))
                .map_err(|err| CamError::V4l2Setup(err.to_string()))?
        };

    let format = capture_queue
        .change_format()
        .map_err(|err| CamError::V4l2Setup(err.to_string()))?
        .set_size(config.width as usize, config.height as usize)
        .set_pixelformat(b"MJPG")
        .apply::<v4l2r::Format>()
        .map_err(|err| CamError::V4l2Setup(err.to_string()))?;

    let pixel_fourcc = <[u8; 4]>::from(format.pixelformat);
    let pixel_str = String::from_utf8_lossy(&pixel_fourcc).to_string();
    if format.pixelformat != v4l2r::PixelFormat::from(b"MJPG")
        && format.pixelformat != v4l2r::PixelFormat::from(b"JPEG")
    {
        return Err(CamError::UnsupportedPixelFormat(pixel_str));
    }

    if format.width != config.width || format.height != config.height {
        return Err(CamError::CameraSizeMismatch {
            requested_width: config.width,
            requested_height: config.height,
            actual_width: format.width,
            actual_height: format.height,
        });
    }

    let capture_queue = capture_queue
        .request_buffers::<Vec<MmapHandle>>(CAMERA_BUFFER_COUNT)
        .map_err(|err| CamError::V4l2Setup(err.to_string()))?;

    for _ in 0..capture_queue.num_buffers() {
        let buf = capture_queue
            .try_get_free_buffer()
            .map_err(|err| CamError::V4l2Setup(err.to_string()))?;
        buf.queue()
            .map_err(|err| CamError::V4l2Setup(err.to_string()))?;
    }

    capture_queue
        .stream_on()
        .map_err(|err| CamError::V4l2Setup(err.to_string()))?;

    let mut decoder = TurboJpegDecoder::new()?;
    let mut frame_id: u64 = 0;

    loop {
        if should_shutdown(&shared)? {
            break;
        }

        let dqbuf = match capture_queue.try_dequeue() {
            Ok(buf) => buf,
            Err(IoctlConvertError::IoctlError(DqBufIoctlError::NotReady)) => {
                thread::sleep(Duration::from_millis(2));
                continue;
            }
            Err(IoctlConvertError::IoctlError(DqBufIoctlError::Eos)) => {
                return Err(CamError::EndOfStream);
            }
            Err(err) => {
                return Err(CamError::V4l2Runtime(err.to_string()));
            }
        };

        let bytes_used = *dqbuf.data.get_first_plane().bytesused as usize;
        if bytes_used == 0 {
            drop(dqbuf);
            requeue_capture_buffer(&capture_queue)?;
            continue;
        }

        let mapping = dqbuf
            .get_plane_mapping(0)
            .ok_or(CamError::CaptureMappingFailed)?;
        let bytes_used = bytes_used.min(mapping.len());
        let jpeg_bytes = &mapping.as_ref()[..bytes_used];

        let slot_index = acquire_decode_slot(&shared)?;
        let slot = &slots[slot_index];
        let slot_width = slot.size.width;
        let slot_height = slot.size.height;

        let decode_res = slot.buffer.with_mapped_bytes_mut(|dst, dst_len| {
            let expected = slot.size.total_pixels();
            if dst_len < expected {
                return Err(CamError::JpegDecode(format!(
                    "destination buffer too small: {} < {}",
                    dst_len, expected
                )));
            }
            decoder.decode_gray_into(
                jpeg_bytes,
                slot_width as i32,
                slot_height as i32,
                dst,
                slot_width as i32,
            )
        });

        drop(mapping);
        drop(dqbuf);
        requeue_capture_buffer(&capture_queue)?;

        match decode_res {
            Ok(()) => {
                frame_id = frame_id.wrapping_add(1);
                publish_ready_frame(
                    &shared,
                    ReadyFrame {
                        slot: slot_index,
                        frame_id,
                        timestamp_ns: now_monotonicish_ns(),
                    },
                )?;
            }
            Err(err) => {
                release_slot(&shared, slot_index)?;
                return Err(err);
            }
        }
    }

    if let Ok(mut guard) = shared.state.lock() {
        guard.shutdown = true;
        shared.cv.notify_all();
    }
    Ok(())
}

fn requeue_capture_buffer(
    capture_queue: &v4l2r::device::queue::Queue<
        Capture,
        v4l2r::device::queue::BuffersAllocated<Vec<MmapHandle>>,
    >,
) -> Result<(), CamError> {
    let buf = capture_queue
        .try_get_free_buffer()
        .map_err(|err| CamError::V4l2Runtime(err.to_string()))?;
    buf.queue()
        .map_err(|err| CamError::V4l2Runtime(err.to_string()))?;
    Ok(())
}

fn should_shutdown(shared: &Arc<SharedQueue>) -> Result<bool, CamError> {
    let guard = shared.state.lock().map_err(|_| CamError::WorkerStopped)?;
    Ok(guard.shutdown)
}

fn acquire_decode_slot(shared: &Arc<SharedQueue>) -> Result<usize, CamError> {
    let mut guard = shared.state.lock().map_err(|_| CamError::WorkerStopped)?;
    loop {
        if guard.shutdown {
            return Err(CamError::WorkerStopped);
        }
        if let Some(slot) = guard.free_slots.pop() {
            return Ok(slot);
        }
        if let Some(old_ready) = guard.latest_ready.take() {
            guard.free_slots.push(old_ready.slot);
            continue;
        }
        guard = shared.cv.wait(guard).map_err(|_| CamError::WorkerStopped)?;
    }
}

fn release_slot(shared: &Arc<SharedQueue>, slot: usize) -> Result<(), CamError> {
    let mut guard = shared.state.lock().map_err(|_| CamError::WorkerStopped)?;
    guard.free_slots.push(slot);
    shared.cv.notify_all();
    Ok(())
}

fn publish_ready_frame(shared: &Arc<SharedQueue>, ready: ReadyFrame) -> Result<(), CamError> {
    let mut guard = shared.state.lock().map_err(|_| CamError::WorkerStopped)?;
    if guard.shutdown {
        guard.free_slots.push(ready.slot);
        shared.cv.notify_all();
        return Err(CamError::WorkerStopped);
    }
    if let Some(old) = guard.latest_ready.replace(ready) {
        guard.free_slots.push(old.slot);
    }
    let _ = ready.frame_id;
    let _ = ready.timestamp_ns;
    shared.cv.notify_all();
    Ok(())
}

fn now_monotonicish_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

struct TurboJpegDecoder {
    handle: tjhandle,
}

impl TurboJpegDecoder {
    fn new() -> Result<Self, CamError> {
        let handle = unsafe { tjInitDecompress() };
        if handle.is_null() {
            return Err(CamError::JpegDecode(
                "tjInitDecompress returned null".to_string(),
            ));
        }
        Ok(Self { handle })
    }

    fn decode_gray_into(
        &mut self,
        jpeg: &[u8],
        expected_width: i32,
        expected_height: i32,
        dst: *mut u8,
        pitch: i32,
    ) -> Result<(), CamError> {
        let mut width = 0i32;
        let mut height = 0i32;
        let mut subsamp = 0i32;
        let mut colorspace = 0i32;
        let header_status = unsafe {
            tjDecompressHeader3(
                self.handle,
                jpeg.as_ptr(),
                jpeg.len() as c_ulong,
                &mut width,
                &mut height,
                &mut subsamp,
                &mut colorspace,
            )
        };
        if header_status != 0 {
            return Err(CamError::JpegDecode(self.last_error_string()));
        }
        let _ = subsamp;
        let _ = colorspace;

        if width != expected_width || height != expected_height {
            return Err(CamError::JpegSizeMismatch {
                expected_width: expected_width as u32,
                expected_height: expected_height as u32,
                actual_width: width as u32,
                actual_height: height as u32,
            });
        }

        let flags = (TJFLAG_FASTUPSAMPLE | TJFLAG_FASTDCT) as i32;
        let status = unsafe {
            tjDecompress2(
                self.handle,
                jpeg.as_ptr(),
                jpeg.len() as c_ulong,
                dst,
                expected_width,
                pitch,
                expected_height,
                TJPF_TJPF_GRAY,
                flags,
            )
        };
        if status != 0 {
            return Err(CamError::JpegDecode(self.last_error_string()));
        }
        Ok(())
    }

    fn last_error_string(&self) -> String {
        let ptr = unsafe { tjGetErrorStr2(self.handle) };
        if ptr.is_null() {
            return "unknown turbojpeg error".to_owned();
        }
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
}

impl Drop for TurboJpegDecoder {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let _ = unsafe { tjDestroy(self.handle) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn normalize_size_matches_detector_alignment() {
        let n0 = normalize_size(Size::new(641, 481), Some(2)).map(|s| (s.width, s.height));
        assert_eq!(n0, Some((640, 480)));

        let n1 = normalize_size(Size::new(3, 3), Some(1)).map(|s| (s.width, s.height));
        assert_eq!(n1, None);

        let n2 = normalize_size(Size::new(1280, 720), None).map(|s| (s.width, s.height));
        assert_eq!(n2, Some((1280, 720)));
    }

    #[test]
    fn publish_replaces_stale_latest_frame() {
        let shared = Arc::new(SharedQueue::new());
        publish_ready_frame(
            &shared,
            ReadyFrame {
                slot: 0,
                frame_id: 1,
                timestamp_ns: 1,
            },
        )
        .expect("first publish should succeed");
        publish_ready_frame(
            &shared,
            ReadyFrame {
                slot: 1,
                frame_id: 2,
                timestamp_ns: 2,
            },
        )
        .expect("second publish should succeed");

        let guard = shared.state.lock().expect("state lock");
        assert_eq!(guard.latest_ready.map(|r| r.slot), Some(1));
        assert!(guard.free_slots.contains(&0));
    }
}
