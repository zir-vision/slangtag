use crate::gpu::{
    BufferMemory, CachedCommandBuffer, CommandRecorder, ComputePipeline, DescriptorBuffer,
    GpuBuffer,
};
use crate::sort::RadixSorter;
use crate::{ComputeDevice, Size, compute_shader_path, include_u32};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use image::{DynamicImage, GrayImage, ImageBuffer};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

include!("apriltag36h11_codes.rs");

#[derive(Clone)]
struct WriteDescriptorSet {
    binding: u32,
    buffer: DescriptorBuffer,
}

impl WriteDescriptorSet {
    fn buffer<T: Pod + Copy>(binding: u32, buffer: GpuBuffer<T>) -> Self {
        Self {
            binding,
            buffer: buffer.descriptor(),
        }
    }
}

#[derive(Clone)]
struct DescriptorSet {
    writes: Vec<WriteDescriptorSet>,
}

#[derive(Copy, Clone)]
struct RotatedTagCode {
    code: u64,
    id: u32,
    rotation: u8,
}

static APRILTAG_36H11_ROTATED_CODES: LazyLock<Vec<RotatedTagCode>> = LazyLock::new(|| {
    let mut rotated_codes = Vec::with_capacity(APRILTAG_36H11_CODES.len() * 4);
    for (id, code) in APRILTAG_36H11_CODES.iter().copied().enumerate() {
        let mut rotated_code = code;
        for rotation in 0..4u8 {
            rotated_codes.push(RotatedTagCode {
                code: rotated_code,
                id: id as u32,
                rotation,
            });
            rotated_code = Detector::rotate_code_ccw(rotated_code, Detector::APRILTAG_MARKER_SIZE);
        }
    }
    rotated_codes
});

static APRILTAG_36H11_EXACT_LOOKUP: LazyLock<HashMap<u64, (u32, u8)>> = LazyLock::new(|| {
    let mut by_code = HashMap::with_capacity(APRILTAG_36H11_ROTATED_CODES.len());
    for entry in APRILTAG_36H11_ROTATED_CODES.iter().copied() {
        by_code
            .entry(entry.code)
            .or_insert((entry.id, entry.rotation));
    }
    by_code
});

pub struct DetectionSettings {
    pub decimate: Option<u8>,
    pub min_white_black_diff: u8,
    pub min_blob_size: u32,
    pub blob_pair_filter: BlobPairFilterSettings,
    pub quad_fit: QuadFitSettings,
    pub decode: DecodeSettings,
    pub apriltag: AprilTagSettings,
}

#[derive(Clone, Copy)]
pub struct BlobPairFilterSettings {
    pub min_tag_width: u32,
    pub tag_width: u32,
    pub reversed_border: u32,
    pub normal_border: u32,
    pub min_cluster_pixels: u32,
    pub max_cluster_pixels: Option<u32>,
    pub max_cluster_pixels_perimeter_scale: u32,
}

#[derive(Clone, Copy)]
pub struct QuadFitSettings {
    pub max_nmaxima: u32,
    pub max_line_fit_mse: f32,
    pub cos_critical_rad: f32,
}

#[derive(Clone, Copy)]
pub struct DecodeSettings {
    pub cell_size: u32,
    pub min_stddev_otsu: f32,
    pub cell_margin_pixels: u32,
    pub cell_span: u32,
    pub detect_inverted_marker: bool,
    pub max_erroneous_border_bits_rate: f32,
}

#[derive(Clone, Copy)]
pub struct AprilTagSettings {
    pub error_correction_rate: f32,
    pub max_correction_bits: u32,
}

#[derive(Debug, Clone)]
pub struct DetectedTag {
    pub quad_index: u32,
    pub id: Option<u32>,
    pub blob_index: u32,
    pub reversed_border: bool,
    pub score: f32,
    pub corners: [[f32; 2]; 4],
    pub bits_with_border: Vec<u8>,
    pub payload_bits: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct GpuTimingSpan {
    pub name: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Clone, Default)]
pub struct GpuTimingReport {
    pub spans: Vec<GpuTimingSpan>,
    pub total_ms: f64,
}

pub struct Detector {
    device: ComputeDevice,
    settings: DetectionSettings,
    pipelines: DetectionPipelines,
    sorter: RadixSorter,
    fixed_size: Option<Size>,
    cached_execution: Mutex<Option<CachedDetectorExecution>>,
}

struct CachedDetectorExecution {
    max_quad_capacity_u32: u32,
    fitted_quad_words_per_quad: usize,
    readback_counters: GpuBuffer<u32>,
    readback_fitted_quads: GpuBuffer<u32>,
    readback_bits: GpuBuffer<u32>,
    _buffers: CachedDetectorBuffers,
    timings: PipelineTimings,
    command_buffer: CachedCommandBuffer,
}

struct CachedDetectorBuffers {
    input_placeholder: GpuBuffer<u8>,
    decimated_image: Option<GpuBuffer<u8>>,
    thresholded_image: Option<GpuBuffer<u8>>,
    minmax_image: GpuBuffer<u8>,
    filtered_minmax_image: GpuBuffer<u8>,
    labels: GpuBuffer<u32>,
    final_labels: GpuBuffer<u32>,
    union_markers_size: GpuBuffer<u32>,
    blob_diff_out: GpuBuffer<u32>,
    blob_diff_compacted: GpuBuffer<u32>,
    blob_diff_sorted: GpuBuffer<u32>,
    blob_extent: GpuBuffer<u32>,
    filtered_blob_extent: GpuBuffer<u32>,
    point_extent_indices: GpuBuffer<u32>,
    selected_blob_points: GpuBuffer<u32>,
    selected_blob_sorted_points: GpuBuffer<u32>,
    line_fit_points: GpuBuffer<u32>,
    errs: GpuBuffer<u32>,
    filtered_errs: GpuBuffer<u32>,
    peaks: GpuBuffer<u32>,
    sorted_peaks: GpuBuffer<u32>,
    peak_extents: GpuBuffer<u32>,
    fitted_quads: GpuBuffer<u32>,
    quad_params: GpuBuffer<u32>,
    bits: GpuBuffer<u32>,
    control: GpuBuffer<u32>,
    blob_sort_keys: GpuBuffer<u32>,
    blob_sorted_indices: GpuBuffer<u32>,
    blob_sort_storage: GpuBuffer<u32>,
    selected_sort_keys: GpuBuffer<u32>,
    selected_sorted_indices: GpuBuffer<u32>,
    selected_sort_storage: GpuBuffer<u32>,
    peak_sort_keys: GpuBuffer<u32>,
    peak_sorted_indices: GpuBuffer<u32>,
    peak_sort_storage: GpuBuffer<u32>,
}

struct DetectionPipelines {
    decimate: Arc<ComputePipeline>,
    minmax: Arc<ComputePipeline>,
    filter_minmax: Arc<ComputePipeline>,
    threshold: Arc<ComputePipeline>,
    ccl_init: Arc<ComputePipeline>,
    ccl_compression: Arc<ComputePipeline>,
    ccl_merge: Arc<ComputePipeline>,
    ccl_final_labeling: Arc<ComputePipeline>,
    blob_diff: Arc<ComputePipeline>,
    filter_nonzero_blob_diff_points: Arc<ComputePipeline>,
    radix_init_keys_indices: Arc<ComputePipeline>,
    radix_keys_from_indices: Arc<ComputePipeline>,
    radix_gather_by_indices: Arc<ComputePipeline>,
    build_blob_pair_extents: Arc<ComputePipeline>,
    rewrite_selected_blob_points_with_theta: Arc<ComputePipeline>,
    rebuild_filtered_extent_starts: Arc<ComputePipeline>,
    build_line_fit_points: Arc<ComputePipeline>,
    fit_line_errors_and_peaks: Arc<ComputePipeline>,
    build_peak_extents: Arc<ComputePipeline>,
    fit_quads: Arc<ComputePipeline>,
    build_indirect_dispatch_args: Arc<ComputePipeline>,
    prepare_decode_quads: Arc<ComputePipeline>,
    extract_candidate_bits: Arc<ComputePipeline>,
}

impl DetectionPipelines {
    fn new(device: &ComputeDevice) -> Self {
        Self {
            decimate: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!("threshold-decimate")),
            ),
            minmax: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!("threshold-minmax")),
            ),
            filter_minmax: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!("threshold-filter-minmax")),
            ),
            threshold: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!("threshold-threshold")),
            ),
            ccl_init: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!("ccl-init")),
            ),
            ccl_compression: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!("ccl-compression")),
            ),
            ccl_merge: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!("ccl-merge")),
            ),
            ccl_final_labeling: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!("ccl-final-labeling")),
            ),
            blob_diff: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!("blob-blob-diff")),
            ),
            filter_nonzero_blob_diff_points: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!(
                    "select-filter-nonzero-blob-diff-points"
                )),
            ),
            radix_init_keys_indices: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!("sort-radix-init-keys-indices")),
            ),
            radix_keys_from_indices: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!("sort-radix-keys-from-indices")),
            ),
            radix_gather_by_indices: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!("sort-radix-gather-by-indices")),
            ),
            build_blob_pair_extents: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!("filter-build-blob-pair-extents")),
            ),
            rewrite_selected_blob_points_with_theta: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!(
                    "filter-rewrite-selected-blob-points-with-theta"
                )),
            ),
            rebuild_filtered_extent_starts: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!(
                    "filter-rebuild-filtered-extent-starts"
                )),
            ),
            build_line_fit_points: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!("filter-build-line-fit-points")),
            ),
            fit_line_errors_and_peaks: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!("filter-fit-line-errors-and-peaks")),
            ),
            build_peak_extents: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!("filter-build-peak-extents")),
            ),
            fit_quads: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!("filter-fit-quads")),
            ),
            build_indirect_dispatch_args: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!("detect-build-indirect-dispatch-args")),
            ),
            prepare_decode_quads: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!("decode-prepare-decode-quads")),
            ),
            extract_candidate_bits: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!("decode-extract-candidate-bits")),
            ),
        }
    }
}

impl Default for DetectionSettings {
    fn default() -> Self {
        Self {
            decimate: Some(2),
            min_white_black_diff: 25,
            min_blob_size: 25,
            blob_pair_filter: BlobPairFilterSettings::default(),
            quad_fit: QuadFitSettings::default(),
            decode: DecodeSettings::default(),
            apriltag: AprilTagSettings::default(),
        }
    }
}

impl Default for BlobPairFilterSettings {
    fn default() -> Self {
        Self {
            min_tag_width: 3,
            tag_width: 1,
            reversed_border: 1,
            normal_border: 1,
            min_cluster_pixels: 24,
            max_cluster_pixels: None,
            max_cluster_pixels_perimeter_scale: 4,
        }
    }
}

impl Default for QuadFitSettings {
    fn default() -> Self {
        Self {
            max_nmaxima: 10,
            max_line_fit_mse: 10.0,
            cos_critical_rad: 0.984_807_73,
        }
    }
}

impl Default for DecodeSettings {
    fn default() -> Self {
        Self {
            cell_size: 4,
            min_stddev_otsu: 5.0,
            cell_margin_pixels: 0,
            cell_span: 4,
            detect_inverted_marker: true,
            max_erroneous_border_bits_rate: 0.35,
        }
    }
}

impl Default for AprilTagSettings {
    fn default() -> Self {
        Self {
            error_correction_rate: 0.6,
            max_correction_bits: 0,
        }
    }
}

fn is_power_of_two(n: u8) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

fn dispatch_groups_1d(total_invocations: u32, local_size_x: u32) -> u32 {
    if total_invocations == 0 {
        1
    } else {
        total_invocations.div_ceil(local_size_x)
    }
}

fn dispatch_groups_2d(width: u32, height: u32, local_size_x: u32, local_size_y: u32) -> [u32; 2] {
    let x = if width == 0 {
        1
    } else {
        width.div_ceil(local_size_x)
    };
    let y = if height == 0 {
        1
    } else {
        height.div_ceil(local_size_y)
    };
    [x, y]
}

struct TimedPass {
    name: String,
    start_query: u32,
    end_query: u32,
}

struct PipelineTimings {
    query_pool: crate::GpuQueryPool,
    next_query: u32,
    spans: Vec<TimedPass>,
}

impl PipelineTimings {
    fn new(device: &ComputeDevice, query_capacity: u32) -> Self {
        Self {
            query_pool: device.create_timestamp_query_pool(query_capacity),
            next_query: 0,
            spans: Vec::new(),
        }
    }

    fn reset(&mut self, commands: &mut CommandRecorder<'_>) {
        commands.reset_query_pool(&self.query_pool, 0, self.query_pool.query_count());
        self.next_query = 0;
        self.spans.clear();
    }

    fn allocate_queries(&mut self, count: u32) -> u32 {
        let base = self.next_query;
        let end = base
            .checked_add(count)
            .expect("timing query index overflow in detect pipeline");
        assert!(
            end <= self.query_pool.query_count(),
            "detect timing query pool exhausted: required={} capacity={}",
            end,
            self.query_pool.query_count()
        );
        self.next_query = end;
        base
    }

    fn dispatch_with_push_constants<T: Pod + Copy>(
        &mut self,
        commands: &mut CommandRecorder<'_>,
        name: &str,
        compute_pipeline: &ComputePipeline,
        bindings: &[(u32, DescriptorBuffer)],
        push_constants: &T,
        dispatch: [u32; 3],
    ) {
        let start_query = self.allocate_queries(1);
        let end_query = self.allocate_queries(1);
        commands.write_timestamp(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            &self.query_pool,
            start_query,
        );
        commands.dispatch_with_push_constants(compute_pipeline, bindings, push_constants, dispatch);
        commands.write_timestamp(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            &self.query_pool,
            end_query,
        );
        self.spans.push(TimedPass {
            name: name.to_owned(),
            start_query,
            end_query,
        });
    }

    fn dispatch_with_push_constants_capture_set<T: Pod + Copy>(
        &mut self,
        commands: &mut CommandRecorder<'_>,
        name: &str,
        compute_pipeline: &ComputePipeline,
        bindings: &[(u32, DescriptorBuffer)],
        push_constants: &T,
        dispatch: [u32; 3],
    ) -> vk::DescriptorSet {
        let start_query = self.allocate_queries(1);
        let end_query = self.allocate_queries(1);
        commands.write_timestamp(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            &self.query_pool,
            start_query,
        );
        let descriptor_set = commands.dispatch_with_push_constants_capture_set(
            compute_pipeline,
            bindings,
            push_constants,
            dispatch,
        );
        commands.write_timestamp(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            &self.query_pool,
            end_query,
        );
        self.spans.push(TimedPass {
            name: name.to_owned(),
            start_query,
            end_query,
        });
        descriptor_set
    }

    fn dispatch_indirect_with_push_constants<T: Pod + Copy>(
        &mut self,
        commands: &mut CommandRecorder<'_>,
        name: &str,
        compute_pipeline: &ComputePipeline,
        bindings: &[(u32, DescriptorBuffer)],
        push_constants: &T,
        indirect_buffer: &GpuBuffer<u32>,
        indirect_offset: vk::DeviceSize,
    ) {
        let start_query = self.allocate_queries(1);
        let end_query = self.allocate_queries(1);
        commands.write_timestamp(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            &self.query_pool,
            start_query,
        );
        commands.dispatch_indirect_with_push_constants(
            compute_pipeline,
            bindings,
            push_constants,
            indirect_buffer,
            indirect_offset,
        );
        commands.write_timestamp(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            &self.query_pool,
            end_query,
        );
        self.spans.push(TimedPass {
            name: name.to_owned(),
            start_query,
            end_query,
        });
    }

    fn dispatch_indirect_with_push_constants_capture_set<T: Pod + Copy>(
        &mut self,
        commands: &mut CommandRecorder<'_>,
        name: &str,
        compute_pipeline: &ComputePipeline,
        bindings: &[(u32, DescriptorBuffer)],
        push_constants: &T,
        indirect_buffer: &GpuBuffer<u32>,
        indirect_offset: vk::DeviceSize,
    ) -> vk::DescriptorSet {
        let start_query = self.allocate_queries(1);
        let end_query = self.allocate_queries(1);
        commands.write_timestamp(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            &self.query_pool,
            start_query,
        );
        let descriptor_set = commands.dispatch_indirect_with_push_constants_capture_set(
            compute_pipeline,
            bindings,
            push_constants,
            indirect_buffer,
            indirect_offset,
        );
        commands.write_timestamp(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            &self.query_pool,
            end_query,
        );
        self.spans.push(TimedPass {
            name: name.to_owned(),
            start_query,
            end_query,
        });
        descriptor_set
    }

    fn reserve_radix_sort_queries(&mut self, name_prefix: &str) -> u32 {
        let base_query = self.allocate_queries(RadixSorter::TIMESTAMP_QUERY_COUNT);
        self.spans.push(TimedPass {
            name: name_prefix.to_owned(),
            start_query: base_query,
            end_query: base_query + (RadixSorter::TIMESTAMP_QUERY_COUNT - 1),
        });
        base_query
    }

    fn summary(&self, device: &ComputeDevice) -> GpuTimingReport {
        if self.next_query == 0 {
            return GpuTimingReport::default();
        }

        let timestamps = device.get_query_pool_results_u64(&self.query_pool, 0, self.next_query);
        let timestamp_period_ns = f64::from(device.timestamp_period_ns());

        let mut spans = Vec::with_capacity(self.spans.len());
        let mut total_ms = 0.0f64;
        for span in &self.spans {
            let start = timestamps[span.start_query as usize];
            let end = timestamps[span.end_query as usize];
            if end < start {
                continue;
            }
            let elapsed_ms = ((end - start) as f64) * timestamp_period_ns / 1_000_000.0;
            total_ms += elapsed_ms;
            spans.push(GpuTimingSpan {
                name: span.name.clone(),
                elapsed_ms,
            });
        }
        GpuTimingReport { spans, total_ms }
    }
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct DecimatePushConstants {
    input_size: Size,
    decimated_size: Size,
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct MinmaxPushConstants {
    input_size: Size,
    minmax_size: Size,
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct FilterMinmaxPushConstants {
    minmax_size: Size,
    filtered_size: Size,
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct ThresholdPushConstants {
    decimated_size: Size,
    filtered_size: Size,
    thresholded_size: Size,
    min_white_black_diff: u32,
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct CclPushConstants {
    image_size: Size,
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct BlobDiffPushConstants {
    thresholded_size: Size,
    min_blob_size: u32,
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct TotalPointsPushConstants {
    total_points: u32,
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct BuildBlobPairExtentsPushConstants {
    valid_points: u32,
    tag_width: u32,
    reversed_border: u32,
    normal_border: u32,
    min_cluster_pixels: u32,
    max_cluster_pixels: u32,
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct RadixExtractPushConstants {
    valid_points: u32,
    words_per_record: u32,
    key_word_index: u32,
    key_transform: u32,
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct RadixGatherPushConstants {
    valid_points: u32,
    words_per_record: u32,
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct RewriteSelectedBlobPointsPushConstants {
    extent_count: u32,
    valid_points: u32,
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct RebuildFilteredExtentStartsPushConstants {
    extent_count: u32,
    point_count: u32,
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct BuildLineFitPointsPushConstants {
    decimated_size: Size,
    extent_count: u32,
    point_count: u32,
    decimate: u32,
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct FitLineErrorsAndPeaksPushConstants {
    extent_count: u32,
    point_count: u32,
    pass: u32,
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct FitQuadsPushConstants {
    peak_extent_count: u32,
    filtered_blob_extent_count: u32,
    max_nmaxima: u32,
    max_line_fit_mse: f32,
    cos_critical_rad: f32,
    min_tag_width: u32,
    quad_decimate: f32,
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct PrepareDecodeQuadsPushConstants {
    image_size: Size,
    quad_count: u32,
    marker_size_with_borders: u32,
    cell_size: u32,
    min_stddev_otsu: f32,
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct ExtractCandidateBitsPushConstants {
    image_size: Size,
    quad_count: u32,
    marker_size_with_borders: u32,
    cell_size: u32,
    cell_margin_pixels: u32,
    cell_span: u32,
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct BuildIndirectDispatchArgsPushConstants {
    one_d_local_size_x: u32,
    marker_size_with_borders: u32,
}

#[derive(Clone)]
struct GpuImageView {
    descriptor: DescriptorBuffer,
    size: Size,
    _keepalive: Option<GpuBuffer<u8>>,
}

impl GpuImageView {
    fn borrowed(descriptor: DescriptorBuffer, size: Size) -> Self {
        Self {
            descriptor,
            size,
            _keepalive: None,
        }
    }

    fn owned(buffer: GpuBuffer<u8>, size: Size) -> Self {
        Self {
            descriptor: buffer.descriptor(),
            size,
            _keepalive: Some(buffer),
        }
    }
}

fn crop_image_to_multiple(image: GrayImage, multiple: u32) -> Result<GrayImage, ()> {
    let width = image.width();
    let height = image.height();
    let cropped_width = width - (width % multiple);
    let cropped_height = height - (height % multiple);

    if cropped_width == 0 || cropped_height == 0 {
        return Err(());
    }

    if cropped_width == width && cropped_height == height {
        return Ok(image);
    }

    Ok(image::imageops::crop_imm(&image, 0, 0, cropped_width, cropped_height).to_image())
}

impl Detector {
    const ONE_D_LOCAL_SIZE_X: u32 = 256;
    const TWO_D_LOCAL_SIZE_X: u32 = 16;
    const TWO_D_LOCAL_SIZE_Y: u32 = 16;
    const FITTED_QUAD_WORDS_PER_QUAD: usize = 15;
    const FITTED_QUAD_BLOB_INDEX_WORD: usize = 0;
    const FITTED_QUAD_REVERSED_BORDER_WORD: usize = 1;
    const FITTED_QUAD_SCORE_WORD: usize = 2;
    const FITTED_QUAD_CORNERS_START_WORD: usize = 3;
    const MARKER_SIZE_WITH_BORDERS: usize = 8;
    const MARKER_BORDER_BITS: usize = 1;
    const APRILTAG_MARKER_SIZE: usize =
        Self::MARKER_SIZE_WITH_BORDERS - (2 * Self::MARKER_BORDER_BITS);
    const RADIX_KEY_TRANSFORM_NONE: u32 = 0;
    const RADIX_KEY_TRANSFORM_F32_ASC: u32 = 1;
    const CONTROL_BLOB_DIFF_FILTERED_COUNT_WORD: u32 = 0;
    const CONTROL_BLOB_EXTENT_COUNT_WORD: u32 = 1;
    const CONTROL_SELECTED_BLOB_EXTENT_COUNT_WORD: u32 = 2;
    const CONTROL_SELECTED_BLOB_POINT_COUNT_WORD: u32 = 3;
    const CONTROL_PEAK_EXTENT_COUNT_WORD: u32 = 4;
    const CONTROL_FITTED_QUAD_COUNT_WORD: u32 = 5;

    const CONTROL_DISPATCH_BUILD_BLOB_PAIR_EXTENTS_WORD: u32 = 16;
    const CONTROL_DISPATCH_RADIX_BLOB_INIT_WORD: u32 = 19;
    const CONTROL_DISPATCH_RADIX_BLOB_UPDATE_WORD: u32 = 22;
    const CONTROL_DISPATCH_RADIX_BLOB_GATHER_WORD: u32 = 25;
    const CONTROL_DISPATCH_REWRITE_SELECTED_BLOB_POINTS_WORD: u32 = 28;
    const CONTROL_DISPATCH_RADIX_SELECTED_INIT_WORD: u32 = 31;
    const CONTROL_DISPATCH_RADIX_SELECTED_UPDATE_WORD: u32 = 34;
    const CONTROL_DISPATCH_RADIX_SELECTED_GATHER_WORD: u32 = 37;
    const CONTROL_DISPATCH_REBUILD_FILTERED_EXTENT_STARTS_WORD: u32 = 40;
    const CONTROL_DISPATCH_BUILD_LINE_FIT_POINTS_WORD: u32 = 43;
    const CONTROL_DISPATCH_FIT_LINE_PASS0_WORD: u32 = 46;
    const CONTROL_DISPATCH_FIT_LINE_PASS1_WORD: u32 = 49;
    const CONTROL_DISPATCH_FIT_LINE_PASS2_WORD: u32 = 52;
    const CONTROL_DISPATCH_RADIX_PEAKS_INIT_WORD: u32 = 55;
    const CONTROL_DISPATCH_RADIX_PEAKS_UPDATE_WORD: u32 = 58;
    const CONTROL_DISPATCH_RADIX_PEAKS_GATHER_WORD: u32 = 61;
    const CONTROL_DISPATCH_BUILD_PEAK_EXTENTS_WORD: u32 = 64;
    const CONTROL_DISPATCH_FIT_QUADS_WORD: u32 = 67;
    const CONTROL_DISPATCH_PREPARE_DECODE_QUADS_WORD: u32 = 70;
    const CONTROL_DISPATCH_EXTRACT_CANDIDATE_BITS_WORD: u32 = 73;
    const CONTROL_WORD_COUNT: usize = 80;
    const CONTROL_COUNTER_WORD_COUNT: usize = 6;

    pub fn new(device: ComputeDevice, settings: DetectionSettings) -> Self {
        let pipelines = DetectionPipelines::new(&device);
        let sorter = RadixSorter::new(device.clone());
        Self {
            device,
            settings,
            pipelines,
            sorter,
            fixed_size: None,
            cached_execution: Mutex::new(None),
        }
    }

    pub fn new_with_size(device: ComputeDevice, settings: DetectionSettings, size: Size) -> Self {
        let mut detector = Self::new(device, settings);
        let normalized_size = detector
            .normalize_input_size(size)
            .expect("input size is too small after decimate alignment");
        detector.fixed_size = Some(normalized_size);
        detector
            .build_cached_execution(normalized_size)
            .map(|cached| {
                detector.cached_execution = Mutex::new(Some(cached));
            })
            .expect("failed to build cached detector execution");
        detector
    }

    fn ensure_cached_execution<'a>(
        &'a self,
        size: Size,
        guard: &'a mut Option<CachedDetectorExecution>,
    ) -> Result<&'a mut CachedDetectorExecution, ()> {
        let normalized_size = self.normalize_input_size(size).ok_or(())?;
        if let Some(expected_size) = self.fixed_size
            && (expected_size.width != normalized_size.width
                || expected_size.height != normalized_size.height)
        {
            return Err(());
        }

        if guard.is_none() {
            *guard = Some(self.build_cached_execution(normalized_size)?);
        }

        guard.as_mut().ok_or(())
    }

    fn normalize_input_size(&self, size: Size) -> Option<Size> {
        let decimate = self.settings.decimate.unwrap_or(1) as u32;
        let multiple = 4 * decimate;
        let width = size.width - (size.width % multiple);
        let height = size.height - (size.height % multiple);
        (width > 0 && height > 0).then_some(Size::new(width, height))
    }

    fn run_cached_execution(
        &self,
        cached: &mut CachedDetectorExecution,
        input: DescriptorBuffer,
    ) -> Result<(Vec<DetectedTag>, GpuTimingReport), ()> {
        self.device.copy_descriptor_buffer_to_buffer(
            input,
            &cached._buffers.input_placeholder,
            cached._buffers.input_placeholder.byte_size(),
        );
        cached.command_buffer.submit_and_wait();

        let timing_report = cached.timings.summary(&self.device);
        let counters = cached.readback_counters.read(Self::CONTROL_COUNTER_WORD_COUNT);
        let fitted_quad_count_value = counters[Self::CONTROL_FITTED_QUAD_COUNT_WORD as usize]
            .min(cached.max_quad_capacity_u32);
        if fitted_quad_count_value == 0 {
            return Ok((Vec::new(), timing_report));
        }

        let fitted_quad_words = cached
            .readback_fitted_quads
            .read(fitted_quad_count_value as usize * cached.fitted_quad_words_per_quad);
        let bits_words = cached
            .readback_bits
            .read(fitted_quad_count_value as usize * Self::MARKER_SIZE_WITH_BORDERS.pow(2));
        let tags =
            self.build_detected_tags(&fitted_quad_words, fitted_quad_count_value, &bits_words);
        Ok((tags, timing_report))
    }

    fn build_cached_execution(&self, size: Size) -> Result<CachedDetectorExecution, ()> {
        self.validate_settings()?;

        let input_placeholder = self.new_u8_storage_buffer(size.total_pixels().max(1));
        let input_gpu_image = GpuImageView::owned(input_placeholder.clone(), size);
        let decimate_factor = self.settings.decimate.unwrap_or(1) as u32;

        let decimated_image = match self.settings.decimate {
            Some(factor) => {
                let decimated_size = crate::Size::new(
                    input_gpu_image.size.width / factor as u32,
                    input_gpu_image.size.height / factor as u32,
                );
                let decimated_image_buffer =
                    self.new_u8_storage_buffer(decimated_size.total_pixels());
                GpuImageView::owned(decimated_image_buffer, decimated_size)
            }
            None => input_gpu_image.clone(),
        };

        let minmax_size = Size::new(
            decimated_image.size.width / 4,
            decimated_image.size.height / 4,
        );
        let minmax_image = self.new_u8_storage_buffer(minmax_size.total_pixels() * 2);
        let filtered_minmax_image = self.new_u8_storage_buffer(minmax_size.total_pixels() * 2);
        let thresholded_image = GpuImageView::owned(
            self.new_u8_storage_buffer(decimated_image.size.total_pixels()),
            decimated_image.size,
        );
        let labels = self.new_u32_storage_buffer(thresholded_image.size.total_pixels());
        let final_labels = self.new_u32_storage_buffer(thresholded_image.size.total_pixels());
        let union_markers_size =
            self.new_zeroed_u32_storage_buffer(thresholded_image.size.total_pixels());

        let blob_diff_words_per_point = 6usize;
        let blob_diff_points_per_offset = (thresholded_image.size.width as usize - 2)
            * (thresholded_image.size.height as usize - 2);
        let blob_diff_total_points = (blob_diff_points_per_offset * 4) as u32;
        let blob_diff_total_points_capacity = usize::max(1, blob_diff_total_points as usize);
        let blob_diff_out = self
            .new_u32_storage_buffer(blob_diff_total_points_capacity * blob_diff_words_per_point);
        let blob_diff_compacted = self
            .new_u32_storage_buffer(blob_diff_total_points_capacity * blob_diff_words_per_point);
        let blob_diff_sorted = self
            .new_u32_storage_buffer(blob_diff_total_points_capacity * blob_diff_words_per_point);

        let blob_extent_words_per_extent = 11usize;
        let blob_extent_capacity = blob_diff_total_points_capacity;
        let blob_extent =
            self.new_u32_storage_buffer(blob_extent_capacity * blob_extent_words_per_extent);
        let filtered_blob_extent =
            self.new_u32_storage_buffer(blob_extent_capacity * blob_extent_words_per_extent);
        let point_extent_indices = self.new_u32_storage_buffer(blob_diff_total_points_capacity);

        let selected_blob_point_words_per_point = 4usize;
        let selected_blob_point_capacity_u32 = blob_diff_total_points.max(1);
        let selected_blob_point_capacity = usize::max(1, selected_blob_point_capacity_u32 as usize);
        let selected_blob_points = self.new_u32_storage_buffer(
            selected_blob_point_capacity * selected_blob_point_words_per_point,
        );
        let selected_blob_sorted_points = self.new_u32_storage_buffer(
            selected_blob_point_capacity * selected_blob_point_words_per_point,
        );

        let line_fit_point_words_per_point = 10usize;
        let line_fit_points = self
            .new_u32_storage_buffer(selected_blob_point_capacity * line_fit_point_words_per_point);
        let errs = self.new_u32_storage_buffer(selected_blob_point_capacity);
        let filtered_errs = self.new_u32_storage_buffer(selected_blob_point_capacity);
        let peak_words_per_peak = 3usize;
        let peaks = self.new_u32_storage_buffer(selected_blob_point_capacity * peak_words_per_peak);
        let sorted_peaks =
            self.new_u32_storage_buffer(selected_blob_point_capacity * peak_words_per_peak);
        let peak_extent_words_per_extent = 3usize;
        let peak_extents = self
            .new_u32_storage_buffer(selected_blob_point_capacity * peak_extent_words_per_extent);

        let fitted_quad_words_per_quad = 15usize;
        let max_quad_capacity_u32 = selected_blob_point_capacity_u32;
        let max_quad_capacity = selected_blob_point_capacity;
        let fitted_quads =
            self.new_u32_storage_buffer(max_quad_capacity * fitted_quad_words_per_quad);
        let quad_param_words_per_quad = 12usize;
        let quad_params =
            self.new_u32_storage_buffer(max_quad_capacity * quad_param_words_per_quad);
        let bits_count_oversized = max_quad_capacity * 8usize * 8usize;
        let bits = self.new_u32_storage_buffer(bits_count_oversized);

        let control = self.new_u32_control_buffer(Self::CONTROL_WORD_COUNT);
        let control_blob_diff_filtered_desc =
            Self::control_counter_descriptor(&control, Self::CONTROL_BLOB_DIFF_FILTERED_COUNT_WORD);
        let control_blob_extent_desc =
            Self::control_counter_descriptor(&control, Self::CONTROL_BLOB_EXTENT_COUNT_WORD);
        let control_selected_blob_extent_desc = Self::control_counter_descriptor(
            &control,
            Self::CONTROL_SELECTED_BLOB_EXTENT_COUNT_WORD,
        );
        let control_selected_blob_point_desc = Self::control_counter_descriptor(
            &control,
            Self::CONTROL_SELECTED_BLOB_POINT_COUNT_WORD,
        );
        let control_peak_extent_desc =
            Self::control_counter_descriptor(&control, Self::CONTROL_PEAK_EXTENT_COUNT_WORD);
        let control_fitted_quad_desc =
            Self::control_counter_descriptor(&control, Self::CONTROL_FITTED_QUAD_COUNT_WORD);

        let blob_sort_keys = self.new_u32_storage_buffer(blob_diff_total_points_capacity);
        let blob_sorted_indices = self.new_u32_storage_buffer(blob_diff_total_points_capacity);
        let blob_sort_storage = self
            .sorter
            .create_key_value_storage_buffer(blob_diff_total_points.max(1));

        let selected_sort_keys = self.new_u32_storage_buffer(selected_blob_point_capacity);
        let selected_sorted_indices = self.new_u32_storage_buffer(selected_blob_point_capacity);
        let selected_sort_storage = self
            .sorter
            .create_key_value_storage_buffer(selected_blob_point_capacity_u32);

        let peak_sort_keys = self.new_u32_storage_buffer(selected_blob_point_capacity);
        let peak_sorted_indices = self.new_u32_storage_buffer(selected_blob_point_capacity);
        let peak_sort_storage = self
            .sorter
            .create_key_value_storage_buffer(selected_blob_point_capacity_u32);

        let readback_counters = self.device.create_buffer(
            Self::CONTROL_COUNTER_WORD_COUNT,
            vk::BufferUsageFlags::TRANSFER_DST,
            BufferMemory::HostRandomAccess,
        );
        let readback_fitted_quads = self.device.create_buffer(
            max_quad_capacity * fitted_quad_words_per_quad,
            vk::BufferUsageFlags::TRANSFER_DST,
            BufferMemory::HostRandomAccess,
        );
        let readback_bits = self.device.create_buffer(
            bits_count_oversized,
            vk::BufferUsageFlags::TRANSFER_DST,
            BufferMemory::HostRandomAccess,
        );

        let blob_pair_filter = self.settings.blob_pair_filter;
        let min_tag_width = blob_pair_filter.min_tag_width;
        let max_cluster_pixels = blob_pair_filter.max_cluster_pixels.unwrap_or(
            blob_pair_filter.max_cluster_pixels_perimeter_scale
                * (thresholded_image.size.width + thresholded_image.size.height),
        );

        let mut timings = PipelineTimings::new(&self.device, 1024);
        let mut command_buffer = self.device.create_cached_command_buffer();
        command_buffer.record(|commands| {
            timings.reset(commands);
            commands.fill_buffer_u32_range(&control, 0, control.byte_size(), 0);
            commands.fill_buffer_u32_range(&blob_extent, 0, blob_extent.byte_size(), 0);
            commands.fill_buffer_u32_range(
                &filtered_blob_extent,
                0,
                filtered_blob_extent.byte_size(),
                0,
            );
            commands.fill_buffer_u32_range(&peak_extents, 0, peak_extents.byte_size(), 0);
            commands.fill_buffer_u32_range(&fitted_quads, 0, fitted_quads.byte_size(), 0);
            commands.barrier_transfer_write_to_compute_read();

            if self.settings.decimate.is_some() {
                let decimate_dispatch = dispatch_groups_2d(
                    input_gpu_image.size.width,
                    input_gpu_image.size.height,
                    Self::TWO_D_LOCAL_SIZE_X,
                    Self::TWO_D_LOCAL_SIZE_Y,
                );
                self.dispatch_with_push_constants_recorded_timed(
                    &mut timings,
                    "threshold-decimate",
                    commands,
                    &self.pipelines.decimate,
                    &[
                        (0, input_gpu_image.descriptor),
                        (1, decimated_image.descriptor),
                    ],
                    DecimatePushConstants {
                        input_size: input_gpu_image.size,
                        decimated_size: decimated_image.size,
                    },
                    [decimate_dispatch[0], decimate_dispatch[1], 1],
                );
                commands.barrier_shader_write_to_shader_read();
            }

            // The rest of the graph is fixed-size and can stay pre-recorded.
            // Keep using the existing helpers for the bulk of dispatches.
            let minmax_dispatch = dispatch_groups_2d(
                minmax_size.width,
                minmax_size.height,
                Self::TWO_D_LOCAL_SIZE_X,
                Self::TWO_D_LOCAL_SIZE_Y,
            );
            self.dispatch_with_push_constants_recorded_timed(
                &mut timings,
                "threshold-minmax",
                commands,
                &self.pipelines.minmax,
                &[
                    (0, decimated_image.descriptor),
                    (1, minmax_image.descriptor()),
                ],
                MinmaxPushConstants {
                    input_size: decimated_image.size,
                    minmax_size,
                },
                [minmax_dispatch[0], minmax_dispatch[1], 1],
            );
            commands.barrier_shader_write_to_shader_read();

            self.dispatch_with_push_constants_recorded_timed(
                &mut timings,
                "threshold-filter-minmax",
                commands,
                &self.pipelines.filter_minmax,
                &[
                    (0, minmax_image.descriptor()),
                    (1, filtered_minmax_image.descriptor()),
                ],
                FilterMinmaxPushConstants {
                    minmax_size,
                    filtered_size: minmax_size,
                },
                [minmax_dispatch[0], minmax_dispatch[1], 1],
            );
            commands.barrier_shader_write_to_shader_read();

            let threshold_dispatch = dispatch_groups_2d(
                thresholded_image.size.width,
                thresholded_image.size.height,
                Self::TWO_D_LOCAL_SIZE_X,
                Self::TWO_D_LOCAL_SIZE_Y,
            );
            self.dispatch_with_push_constants_recorded_timed(
                &mut timings,
                "threshold-threshold",
                commands,
                &self.pipelines.threshold,
                &[
                    (0, decimated_image.descriptor),
                    (1, filtered_minmax_image.descriptor()),
                    (2, thresholded_image.descriptor),
                ],
                ThresholdPushConstants {
                    decimated_size: decimated_image.size,
                    filtered_size: minmax_size,
                    thresholded_size: decimated_image.size,
                    min_white_black_diff: self.settings.min_white_black_diff as u32,
                },
                [threshold_dispatch[0], threshold_dispatch[1], 1],
            );
            commands.barrier_shader_write_to_shader_read();

            let ccl_dispatch = dispatch_groups_2d(
                thresholded_image.size.width / 2,
                thresholded_image.size.height / 2,
                Self::TWO_D_LOCAL_SIZE_X,
                Self::TWO_D_LOCAL_SIZE_Y,
            );
            self.dispatch_with_push_constants_recorded_timed(
                &mut timings,
                "ccl-init",
                commands,
                &self.pipelines.ccl_init,
                &[(0, thresholded_image.descriptor), (1, labels.descriptor())],
                CclPushConstants {
                    image_size: thresholded_image.size,
                },
                [ccl_dispatch[0], ccl_dispatch[1], 1],
            );
            commands.barrier_shader_write_to_shader_read();

            self.dispatch_with_push_constants_recorded_timed(
                &mut timings,
                "ccl-compression[first]",
                commands,
                &self.pipelines.ccl_compression,
                &[(0, labels.descriptor())],
                CclPushConstants {
                    image_size: thresholded_image.size,
                },
                [ccl_dispatch[0], ccl_dispatch[1], 1],
            );
            commands.barrier_shader_write_to_shader_read();

            self.dispatch_with_push_constants_recorded_timed(
                &mut timings,
                "ccl-merge",
                commands,
                &self.pipelines.ccl_merge,
                &[(0, labels.descriptor())],
                CclPushConstants {
                    image_size: thresholded_image.size,
                },
                [ccl_dispatch[0], ccl_dispatch[1], 1],
            );
            commands.barrier_shader_write_to_shader_read();

            self.dispatch_with_push_constants_recorded_timed(
                &mut timings,
                "ccl-compression[second]",
                commands,
                &self.pipelines.ccl_compression,
                &[(0, labels.descriptor())],
                CclPushConstants {
                    image_size: thresholded_image.size,
                },
                [ccl_dispatch[0], ccl_dispatch[1], 1],
            );
            commands.barrier_shader_write_to_shader_read();

            self.dispatch_with_push_constants_recorded_timed(
                &mut timings,
                "ccl-final-labeling",
                commands,
                &self.pipelines.ccl_final_labeling,
                &[
                    (0, labels.descriptor()),
                    (1, final_labels.descriptor()),
                    (2, union_markers_size.descriptor()),
                ],
                CclPushConstants {
                    image_size: thresholded_image.size,
                },
                [ccl_dispatch[0], ccl_dispatch[1], 1],
            );
            commands.barrier_shader_write_to_shader_read();

            self.dispatch_with_push_constants_recorded_timed(
                &mut timings,
                "blob-blob-diff",
                commands,
                &self.pipelines.blob_diff,
                &[
                    (0, thresholded_image.descriptor),
                    (1, final_labels.descriptor()),
                    (2, union_markers_size.descriptor()),
                    (3, blob_diff_out.descriptor()),
                ],
                BlobDiffPushConstants {
                    thresholded_size: thresholded_image.size,
                    min_blob_size: self.settings.min_blob_size,
                },
                [threshold_dispatch[0], threshold_dispatch[1], 1],
            );
            commands.barrier_shader_write_to_shader_read();

            self.dispatch_with_push_constants_recorded_timed(
                &mut timings,
                "select-filter-nonzero-blob-diff-points",
                commands,
                &self.pipelines.filter_nonzero_blob_diff_points,
                &[
                    (0, blob_diff_out.descriptor()),
                    (1, blob_diff_compacted.descriptor()),
                    (2, control_blob_diff_filtered_desc),
                ],
                TotalPointsPushConstants {
                    total_points: blob_diff_total_points,
                },
                [
                    dispatch_groups_1d(blob_diff_total_points, Self::ONE_D_LOCAL_SIZE_X),
                    1,
                    1,
                ],
            );
            commands.barrier_shader_write_to_shader_read();

            self.dispatch_with_push_constants_recorded_timed(
                &mut timings,
                "detect-build-indirect-dispatch-args[initial]",
                commands,
                &self.pipelines.build_indirect_dispatch_args,
                &[(0, control.descriptor()), (1, control.descriptor())],
                BuildIndirectDispatchArgsPushConstants {
                    one_d_local_size_x: Self::ONE_D_LOCAL_SIZE_X,
                    marker_size_with_borders: Self::MARKER_SIZE_WITH_BORDERS as u32,
                },
                [1, 1, 1],
            );
            commands.barrier_shader_write_to_shader_read();
            commands.barrier_shader_write_to_indirect_read();

            self.dispatch_indirect_with_push_constants_recorded_timed(
                &mut timings,
                "radix-sort-blob-diff-points::sort-key-init",
                commands,
                &self.pipelines.radix_init_keys_indices,
                &[
                    (0, blob_diff_compacted.descriptor()),
                    (1, blob_sort_keys.descriptor()),
                    (2, blob_sorted_indices.descriptor()),
                ],
                RadixExtractPushConstants {
                    valid_points: blob_diff_total_points,
                    words_per_record: 6,
                    key_word_index: 1,
                    key_transform: Self::RADIX_KEY_TRANSFORM_NONE,
                },
                &control,
                Self::control_dispatch_offset(Self::CONTROL_DISPATCH_RADIX_BLOB_INIT_WORD),
            );
            commands.barrier_shader_write_to_shader_read();

            let secondary_blob_query = timings
                .reserve_radix_sort_queries("radix-sort-blob-diff-points::secondary-key-sort");
            self.sorter.record_sort_key_value_indirect_with_query_pool(
                commands,
                blob_diff_total_points.max(1),
                &control,
                Self::control_word_offset(Self::CONTROL_BLOB_DIFF_FILTERED_COUNT_WORD),
                &blob_sort_keys,
                0,
                &blob_sorted_indices,
                0,
                &blob_sort_storage,
                0,
                Some((&timings.query_pool, secondary_blob_query)),
            );
            commands.barrier_shader_write_to_shader_read();

            self.dispatch_indirect_with_push_constants_recorded_timed(
                &mut timings,
                "radix-sort-blob-diff-points::sort-key-update",
                commands,
                &self.pipelines.radix_keys_from_indices,
                &[
                    (0, blob_diff_compacted.descriptor()),
                    (1, blob_sorted_indices.descriptor()),
                    (2, blob_sort_keys.descriptor()),
                ],
                RadixExtractPushConstants {
                    valid_points: blob_diff_total_points,
                    words_per_record: 6,
                    key_word_index: 0,
                    key_transform: Self::RADIX_KEY_TRANSFORM_NONE,
                },
                &control,
                Self::control_dispatch_offset(Self::CONTROL_DISPATCH_RADIX_BLOB_UPDATE_WORD),
            );
            commands.barrier_shader_write_to_shader_read();

            let primary_blob_query = timings
                .reserve_radix_sort_queries("radix-sort-blob-diff-points::primary-key-sort");
            self.sorter.record_sort_key_value_indirect_with_query_pool(
                commands,
                blob_diff_total_points.max(1),
                &control,
                Self::control_word_offset(Self::CONTROL_BLOB_DIFF_FILTERED_COUNT_WORD),
                &blob_sort_keys,
                0,
                &blob_sorted_indices,
                0,
                &blob_sort_storage,
                0,
                Some((&timings.query_pool, primary_blob_query)),
            );
            commands.barrier_shader_write_to_shader_read();

            self.dispatch_indirect_with_push_constants_recorded_timed(
                &mut timings,
                "radix-sort-blob-diff-points::gather",
                commands,
                &self.pipelines.radix_gather_by_indices,
                &[
                    (0, blob_diff_compacted.descriptor()),
                    (1, blob_sorted_indices.descriptor()),
                    (2, blob_diff_sorted.descriptor()),
                ],
                RadixGatherPushConstants {
                    valid_points: blob_diff_total_points,
                    words_per_record: 6,
                },
                &control,
                Self::control_dispatch_offset(Self::CONTROL_DISPATCH_RADIX_BLOB_GATHER_WORD),
            );
            commands.barrier_shader_write_to_shader_read();

            self.dispatch_indirect_with_push_constants_recorded_timed(
                &mut timings,
                "filter-build-blob-pair-extents",
                commands,
                &self.pipelines.build_blob_pair_extents,
                &[
                    (0, blob_diff_sorted.descriptor()),
                    (1, blob_extent.descriptor()),
                    (2, control_blob_extent_desc),
                    (3, filtered_blob_extent.descriptor()),
                    (4, control_selected_blob_extent_desc),
                    (5, control_selected_blob_point_desc),
                    (6, point_extent_indices.descriptor()),
                    (7, control.descriptor()),
                ],
                BuildBlobPairExtentsPushConstants {
                    valid_points: blob_diff_total_points,
                    tag_width: blob_pair_filter.tag_width,
                    reversed_border: blob_pair_filter.reversed_border,
                    normal_border: blob_pair_filter.normal_border,
                    min_cluster_pixels: blob_pair_filter.min_cluster_pixels,
                    max_cluster_pixels,
                },
                &control,
                Self::control_dispatch_offset(Self::CONTROL_DISPATCH_BUILD_BLOB_PAIR_EXTENTS_WORD),
            );
            commands.barrier_shader_write_to_shader_read();

            self.dispatch_with_push_constants_recorded_timed(
                &mut timings,
                "detect-build-indirect-dispatch-args[post-extents]",
                commands,
                &self.pipelines.build_indirect_dispatch_args,
                &[(0, control.descriptor()), (1, control.descriptor())],
                BuildIndirectDispatchArgsPushConstants {
                    one_d_local_size_x: Self::ONE_D_LOCAL_SIZE_X,
                    marker_size_with_borders: Self::MARKER_SIZE_WITH_BORDERS as u32,
                },
                [1, 1, 1],
            );
            commands.barrier_shader_write_to_shader_read();
            commands.barrier_shader_write_to_indirect_read();

            self.dispatch_indirect_with_push_constants_recorded_timed(
                &mut timings,
                "filter-rewrite-selected-blob-points-with-theta",
                commands,
                &self.pipelines.rewrite_selected_blob_points_with_theta,
                &[
                    (0, blob_diff_sorted.descriptor()),
                    (1, blob_extent.descriptor()),
                    (2, point_extent_indices.descriptor()),
                    (3, filtered_blob_extent.descriptor()),
                    (4, selected_blob_points.descriptor()),
                    (5, control.descriptor()),
                ],
                RewriteSelectedBlobPointsPushConstants {
                    extent_count: blob_diff_total_points,
                    valid_points: blob_diff_total_points,
                },
                &control,
                Self::control_dispatch_offset(
                    Self::CONTROL_DISPATCH_REWRITE_SELECTED_BLOB_POINTS_WORD,
                ),
            );
            commands.barrier_shader_write_to_shader_read();

            self.dispatch_indirect_with_push_constants_recorded_timed(
                &mut timings,
                "radix-sort-selected-blob-points::sort-key-init",
                commands,
                &self.pipelines.radix_init_keys_indices,
                &[
                    (0, selected_blob_points.descriptor()),
                    (1, selected_sort_keys.descriptor()),
                    (2, selected_sorted_indices.descriptor()),
                ],
                RadixExtractPushConstants {
                    valid_points: selected_blob_point_capacity_u32,
                    words_per_record: 4,
                    key_word_index: 1,
                    key_transform: Self::RADIX_KEY_TRANSFORM_NONE,
                },
                &control,
                Self::control_dispatch_offset(Self::CONTROL_DISPATCH_RADIX_SELECTED_INIT_WORD),
            );
            commands.barrier_shader_write_to_shader_read();

            let secondary_selected_query = timings
                .reserve_radix_sort_queries("radix-sort-selected-blob-points::secondary-key-sort");
            self.sorter.record_sort_key_value_indirect_with_query_pool(
                commands,
                selected_blob_point_capacity_u32,
                &control,
                Self::control_word_offset(Self::CONTROL_SELECTED_BLOB_POINT_COUNT_WORD),
                &selected_sort_keys,
                0,
                &selected_sorted_indices,
                0,
                &selected_sort_storage,
                0,
                Some((&timings.query_pool, secondary_selected_query)),
            );
            commands.barrier_shader_write_to_shader_read();

            self.dispatch_indirect_with_push_constants_recorded_timed(
                &mut timings,
                "radix-sort-selected-blob-points::sort-key-update",
                commands,
                &self.pipelines.radix_keys_from_indices,
                &[
                    (0, selected_blob_points.descriptor()),
                    (1, selected_sorted_indices.descriptor()),
                    (2, selected_sort_keys.descriptor()),
                ],
                RadixExtractPushConstants {
                    valid_points: selected_blob_point_capacity_u32,
                    words_per_record: 4,
                    key_word_index: 0,
                    key_transform: Self::RADIX_KEY_TRANSFORM_NONE,
                },
                &control,
                Self::control_dispatch_offset(Self::CONTROL_DISPATCH_RADIX_SELECTED_UPDATE_WORD),
            );
            commands.barrier_shader_write_to_shader_read();

            let primary_selected_query = timings
                .reserve_radix_sort_queries("radix-sort-selected-blob-points::primary-key-sort");
            self.sorter.record_sort_key_value_indirect_with_query_pool(
                commands,
                selected_blob_point_capacity_u32,
                &control,
                Self::control_word_offset(Self::CONTROL_SELECTED_BLOB_POINT_COUNT_WORD),
                &selected_sort_keys,
                0,
                &selected_sorted_indices,
                0,
                &selected_sort_storage,
                0,
                Some((&timings.query_pool, primary_selected_query)),
            );
            commands.barrier_shader_write_to_shader_read();

            self.dispatch_indirect_with_push_constants_recorded_timed(
                &mut timings,
                "radix-sort-selected-blob-points::gather",
                commands,
                &self.pipelines.radix_gather_by_indices,
                &[
                    (0, selected_blob_points.descriptor()),
                    (1, selected_sorted_indices.descriptor()),
                    (2, selected_blob_sorted_points.descriptor()),
                ],
                RadixGatherPushConstants {
                    valid_points: selected_blob_point_capacity_u32,
                    words_per_record: 4,
                },
                &control,
                Self::control_dispatch_offset(Self::CONTROL_DISPATCH_RADIX_SELECTED_GATHER_WORD),
            );
            commands.barrier_shader_write_to_shader_read();

            self.dispatch_indirect_with_push_constants_recorded_timed(
                &mut timings,
                "filter-rebuild-filtered-extent-starts",
                commands,
                &self.pipelines.rebuild_filtered_extent_starts,
                &[
                    (0, selected_blob_sorted_points.descriptor()),
                    (1, filtered_blob_extent.descriptor()),
                    (2, control.descriptor()),
                ],
                RebuildFilteredExtentStartsPushConstants {
                    extent_count: blob_diff_total_points,
                    point_count: selected_blob_point_capacity_u32,
                },
                &control,
                Self::control_dispatch_offset(
                    Self::CONTROL_DISPATCH_REBUILD_FILTERED_EXTENT_STARTS_WORD,
                ),
            );
            commands.barrier_shader_write_to_shader_read();

            self.dispatch_indirect_with_push_constants_recorded_timed(
                &mut timings,
                "filter-build-line-fit-points",
                commands,
                &self.pipelines.build_line_fit_points,
                &[
                    (0, selected_blob_sorted_points.descriptor()),
                    (1, decimated_image.descriptor),
                    (2, filtered_blob_extent.descriptor()),
                    (3, line_fit_points.descriptor()),
                    (4, control.descriptor()),
                ],
                BuildLineFitPointsPushConstants {
                    decimated_size: decimated_image.size,
                    extent_count: blob_diff_total_points,
                    point_count: selected_blob_point_capacity_u32,
                    decimate: decimate_factor,
                },
                &control,
                Self::control_dispatch_offset(Self::CONTROL_DISPATCH_BUILD_LINE_FIT_POINTS_WORD),
            );
            commands.barrier_shader_write_to_shader_read();

            self.dispatch_indirect_with_push_constants_recorded_timed(
                &mut timings,
                "filter-fit-line-errors-and-peaks",
                commands,
                &self.pipelines.fit_line_errors_and_peaks,
                &[
                    (0, line_fit_points.descriptor()),
                    (1, filtered_blob_extent.descriptor()),
                    (2, errs.descriptor()),
                    (3, filtered_errs.descriptor()),
                    (4, peaks.descriptor()),
                    (5, control.descriptor()),
                ],
                FitLineErrorsAndPeaksPushConstants {
                    extent_count: blob_diff_total_points,
                    point_count: selected_blob_point_capacity_u32,
                    pass: 0,
                },
                &control,
                Self::control_dispatch_offset(Self::CONTROL_DISPATCH_FIT_LINE_PASS0_WORD),
            );
            commands.barrier_shader_write_to_shader_read();

            self.dispatch_indirect_with_push_constants_recorded_timed(
                &mut timings,
                "filter-fit-line-errors-and-peaks[filter]",
                commands,
                &self.pipelines.fit_line_errors_and_peaks,
                &[
                    (0, line_fit_points.descriptor()),
                    (1, filtered_blob_extent.descriptor()),
                    (2, errs.descriptor()),
                    (3, filtered_errs.descriptor()),
                    (4, peaks.descriptor()),
                    (5, control.descriptor()),
                ],
                FitLineErrorsAndPeaksPushConstants {
                    extent_count: blob_diff_total_points,
                    point_count: selected_blob_point_capacity_u32,
                    pass: 1,
                },
                &control,
                Self::control_dispatch_offset(Self::CONTROL_DISPATCH_FIT_LINE_PASS1_WORD),
            );
            commands.barrier_shader_write_to_shader_read();

            self.dispatch_indirect_with_push_constants_recorded_timed(
                &mut timings,
                "filter-fit-line-errors-and-peaks[peaks]",
                commands,
                &self.pipelines.fit_line_errors_and_peaks,
                &[
                    (0, line_fit_points.descriptor()),
                    (1, filtered_blob_extent.descriptor()),
                    (2, errs.descriptor()),
                    (3, filtered_errs.descriptor()),
                    (4, peaks.descriptor()),
                    (5, control.descriptor()),
                ],
                FitLineErrorsAndPeaksPushConstants {
                    extent_count: blob_diff_total_points,
                    point_count: selected_blob_point_capacity_u32,
                    pass: 2,
                },
                &control,
                Self::control_dispatch_offset(Self::CONTROL_DISPATCH_FIT_LINE_PASS2_WORD),
            );
            commands.barrier_shader_write_to_shader_read();

            self.dispatch_indirect_with_push_constants_recorded_timed(
                &mut timings,
                "radix-sort-peaks::sort-key-init",
                commands,
                &self.pipelines.radix_init_keys_indices,
                &[
                    (0, peaks.descriptor()),
                    (1, peak_sort_keys.descriptor()),
                    (2, peak_sorted_indices.descriptor()),
                ],
                RadixExtractPushConstants {
                    valid_points: selected_blob_point_capacity_u32,
                    words_per_record: 3,
                    key_word_index: 1,
                    key_transform: Self::RADIX_KEY_TRANSFORM_F32_ASC,
                },
                &control,
                Self::control_dispatch_offset(Self::CONTROL_DISPATCH_RADIX_PEAKS_INIT_WORD),
            );
            commands.barrier_shader_write_to_shader_read();

            let secondary_peak_query =
                timings.reserve_radix_sort_queries("radix-sort-peaks::secondary-key-sort");
            self.sorter.record_sort_key_value_indirect_with_query_pool(
                commands,
                selected_blob_point_capacity_u32,
                &control,
                Self::control_word_offset(Self::CONTROL_SELECTED_BLOB_POINT_COUNT_WORD),
                &peak_sort_keys,
                0,
                &peak_sorted_indices,
                0,
                &peak_sort_storage,
                0,
                Some((&timings.query_pool, secondary_peak_query)),
            );
            commands.barrier_shader_write_to_shader_read();

            self.dispatch_indirect_with_push_constants_recorded_timed(
                &mut timings,
                "radix-sort-peaks::sort-key-update",
                commands,
                &self.pipelines.radix_keys_from_indices,
                &[
                    (0, peaks.descriptor()),
                    (1, peak_sorted_indices.descriptor()),
                    (2, peak_sort_keys.descriptor()),
                ],
                RadixExtractPushConstants {
                    valid_points: selected_blob_point_capacity_u32,
                    words_per_record: 3,
                    key_word_index: 0,
                    key_transform: Self::RADIX_KEY_TRANSFORM_NONE,
                },
                &control,
                Self::control_dispatch_offset(Self::CONTROL_DISPATCH_RADIX_PEAKS_UPDATE_WORD),
            );
            commands.barrier_shader_write_to_shader_read();

            let primary_peak_query =
                timings.reserve_radix_sort_queries("radix-sort-peaks::primary-key-sort");
            self.sorter.record_sort_key_value_indirect_with_query_pool(
                commands,
                selected_blob_point_capacity_u32,
                &control,
                Self::control_word_offset(Self::CONTROL_SELECTED_BLOB_POINT_COUNT_WORD),
                &peak_sort_keys,
                0,
                &peak_sorted_indices,
                0,
                &peak_sort_storage,
                0,
                Some((&timings.query_pool, primary_peak_query)),
            );
            commands.barrier_shader_write_to_shader_read();

            self.dispatch_indirect_with_push_constants_recorded_timed(
                &mut timings,
                "radix-sort-peaks::gather",
                commands,
                &self.pipelines.radix_gather_by_indices,
                &[
                    (0, peaks.descriptor()),
                    (1, peak_sorted_indices.descriptor()),
                    (2, sorted_peaks.descriptor()),
                ],
                RadixGatherPushConstants {
                    valid_points: selected_blob_point_capacity_u32,
                    words_per_record: 3,
                },
                &control,
                Self::control_dispatch_offset(Self::CONTROL_DISPATCH_RADIX_PEAKS_GATHER_WORD),
            );
            commands.barrier_shader_write_to_shader_read();

            self.dispatch_indirect_with_push_constants_recorded_timed(
                &mut timings,
                "filter-build-peak-extents",
                commands,
                &self.pipelines.build_peak_extents,
                &[
                    (0, sorted_peaks.descriptor()),
                    (1, peak_extents.descriptor()),
                    (2, control_peak_extent_desc),
                    (3, control.descriptor()),
                ],
                TotalPointsPushConstants {
                    total_points: selected_blob_point_capacity_u32,
                },
                &control,
                Self::control_dispatch_offset(Self::CONTROL_DISPATCH_BUILD_PEAK_EXTENTS_WORD),
            );
            commands.barrier_shader_write_to_shader_read();

            self.dispatch_with_push_constants_recorded_timed(
                &mut timings,
                "detect-build-indirect-dispatch-args[post-peak-extents]",
                commands,
                &self.pipelines.build_indirect_dispatch_args,
                &[(0, control.descriptor()), (1, control.descriptor())],
                BuildIndirectDispatchArgsPushConstants {
                    one_d_local_size_x: Self::ONE_D_LOCAL_SIZE_X,
                    marker_size_with_borders: Self::MARKER_SIZE_WITH_BORDERS as u32,
                },
                [1, 1, 1],
            );
            commands.barrier_shader_write_to_shader_read();
            commands.barrier_shader_write_to_indirect_read();

            self.dispatch_indirect_with_push_constants_recorded_timed(
                &mut timings,
                "filter-fit-quads",
                commands,
                &self.pipelines.fit_quads,
                &[
                    (0, sorted_peaks.descriptor()),
                    (1, peak_extents.descriptor()),
                    (2, line_fit_points.descriptor()),
                    (3, filtered_blob_extent.descriptor()),
                    (4, fitted_quads.descriptor()),
                    (5, control_fitted_quad_desc),
                    (6, control.descriptor()),
                ],
                FitQuadsPushConstants {
                    peak_extent_count: selected_blob_point_capacity_u32,
                    filtered_blob_extent_count: blob_diff_total_points,
                    max_nmaxima: self.settings.quad_fit.max_nmaxima,
                    max_line_fit_mse: self.settings.quad_fit.max_line_fit_mse,
                    cos_critical_rad: self.settings.quad_fit.cos_critical_rad,
                    min_tag_width,
                    quad_decimate: decimate_factor as f32,
                },
                &control,
                Self::control_dispatch_offset(Self::CONTROL_DISPATCH_FIT_QUADS_WORD),
            );
            commands.barrier_shader_write_to_shader_read();

            self.dispatch_with_push_constants_recorded_timed(
                &mut timings,
                "detect-build-indirect-dispatch-args[post-fit-quads]",
                commands,
                &self.pipelines.build_indirect_dispatch_args,
                &[(0, control.descriptor()), (1, control.descriptor())],
                BuildIndirectDispatchArgsPushConstants {
                    one_d_local_size_x: Self::ONE_D_LOCAL_SIZE_X,
                    marker_size_with_borders: Self::MARKER_SIZE_WITH_BORDERS as u32,
                },
                [1, 1, 1],
            );
            commands.barrier_shader_write_to_shader_read();
            commands.barrier_shader_write_to_indirect_read();

            self.dispatch_indirect_with_push_constants_recorded_timed(
                &mut timings,
                "decode-prepare-decode-quads",
                commands,
                &self.pipelines.prepare_decode_quads,
                &[
                    (0, input_gpu_image.descriptor),
                    (1, fitted_quads.descriptor()),
                    (2, quad_params.descriptor()),
                    (3, control.descriptor()),
                ],
                PrepareDecodeQuadsPushConstants {
                    image_size: input_gpu_image.size,
                    quad_count: max_quad_capacity_u32,
                    marker_size_with_borders: Self::MARKER_SIZE_WITH_BORDERS as u32,
                    cell_size: self.settings.decode.cell_size,
                    min_stddev_otsu: self.settings.decode.min_stddev_otsu,
                },
                &control,
                Self::control_dispatch_offset(Self::CONTROL_DISPATCH_PREPARE_DECODE_QUADS_WORD),
            );
            commands.barrier_shader_write_to_shader_read();

            self.dispatch_indirect_with_push_constants_recorded_timed(
                &mut timings,
                "decode-extract-candidate-bits",
                commands,
                &self.pipelines.extract_candidate_bits,
                &[
                    (0, input_gpu_image.descriptor),
                    (1, quad_params.descriptor()),
                    (2, bits.descriptor()),
                    (3, control.descriptor()),
                ],
                ExtractCandidateBitsPushConstants {
                    image_size: input_gpu_image.size,
                    quad_count: max_quad_capacity_u32,
                    marker_size_with_borders: Self::MARKER_SIZE_WITH_BORDERS as u32,
                    cell_size: self.settings.decode.cell_size,
                    cell_margin_pixels: self.settings.decode.cell_margin_pixels,
                    cell_span: self.settings.decode.cell_span,
                },
                &control,
                Self::control_dispatch_offset(Self::CONTROL_DISPATCH_EXTRACT_CANDIDATE_BITS_WORD),
            );

            commands.barrier_shader_write_to_transfer_read();
            commands.copy_buffer_region(
                &control,
                0,
                &readback_counters,
                0,
                (Self::CONTROL_COUNTER_WORD_COUNT * std::mem::size_of::<u32>()) as vk::DeviceSize,
            );
            commands.copy_buffer_region(
                &fitted_quads,
                0,
                &readback_fitted_quads,
                0,
                fitted_quads.byte_size(),
            );
            commands.copy_buffer_region(&bits, 0, &readback_bits, 0, bits.byte_size());
        });

        let buffers = CachedDetectorBuffers {
            input_placeholder,
            decimated_image: decimated_image._keepalive.clone(),
            thresholded_image: thresholded_image._keepalive.clone(),
            minmax_image: minmax_image.clone(),
            filtered_minmax_image: filtered_minmax_image.clone(),
            labels: labels.clone(),
            final_labels: final_labels.clone(),
            union_markers_size: union_markers_size.clone(),
            blob_diff_out: blob_diff_out.clone(),
            blob_diff_compacted: blob_diff_compacted.clone(),
            blob_diff_sorted: blob_diff_sorted.clone(),
            blob_extent: blob_extent.clone(),
            filtered_blob_extent: filtered_blob_extent.clone(),
            point_extent_indices: point_extent_indices.clone(),
            selected_blob_points: selected_blob_points.clone(),
            selected_blob_sorted_points: selected_blob_sorted_points.clone(),
            line_fit_points: line_fit_points.clone(),
            errs: errs.clone(),
            filtered_errs: filtered_errs.clone(),
            peaks: peaks.clone(),
            sorted_peaks: sorted_peaks.clone(),
            peak_extents: peak_extents.clone(),
            fitted_quads: fitted_quads.clone(),
            quad_params: quad_params.clone(),
            bits: bits.clone(),
            control: control.clone(),
            blob_sort_keys: blob_sort_keys.clone(),
            blob_sorted_indices: blob_sorted_indices.clone(),
            blob_sort_storage: blob_sort_storage.clone(),
            selected_sort_keys: selected_sort_keys.clone(),
            selected_sorted_indices: selected_sorted_indices.clone(),
            selected_sort_storage: selected_sort_storage.clone(),
            peak_sort_keys: peak_sort_keys.clone(),
            peak_sorted_indices: peak_sorted_indices.clone(),
            peak_sort_storage: peak_sort_storage.clone(),
        };

        Ok(CachedDetectorExecution {
            max_quad_capacity_u32,
            fitted_quad_words_per_quad,
            readback_counters,
            readback_fitted_quads,
            readback_bits,
            _buffers: buffers,
            timings,
            command_buffer,
        })
    }

    pub fn detect(&self, image: DynamicImage) -> Result<Vec<DetectedTag>, ()> {
        let gray_image = image.to_luma8();
        self.detect_gray(gray_image)
    }

    pub fn detect_gray(
        &self,
        image: ImageBuffer<image::Luma<u8>, Vec<u8>>,
    ) -> Result<Vec<DetectedTag>, ()> {
        self.detect_gray_with_timing(image).map(|(tags, _)| tags)
    }

    pub fn detect_gpu_buffer(
        &self,
        input: DescriptorBuffer,
        size: Size,
    ) -> Result<Vec<DetectedTag>, ()> {
        self.detect_gpu_buffer_with_timing(input, size)
            .map(|(tags, _)| tags)
    }

    pub fn detect_gpu_buffer_with_timing(
        &self,
        input: DescriptorBuffer,
        size: Size,
    ) -> Result<(Vec<DetectedTag>, GpuTimingReport), ()> {
        self.validate_settings()?;
        self.detect_gpu_buffer_with_timing_single_submit(input, size)
    }

    pub fn detect_vk_buffer(
        &self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
        size: Size,
    ) -> Result<Vec<DetectedTag>, ()> {
        self.detect_gpu_buffer(DescriptorBuffer::new(buffer, offset, range), size)
    }

    pub fn detect_vk_buffer_with_timing(
        &self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
        size: Size,
    ) -> Result<(Vec<DetectedTag>, GpuTimingReport), ()> {
        self.detect_gpu_buffer_with_timing(DescriptorBuffer::new(buffer, offset, range), size)
    }

    fn detect_gpu_buffer_with_timing_single_submit(
        &self,
        input: DescriptorBuffer,
        size: Size,
    ) -> Result<(Vec<DetectedTag>, GpuTimingReport), ()> {
        let mut guard = self
            .cached_execution
            .lock()
            .expect("failed to lock cached detector execution");
        let cached = self.ensure_cached_execution(size, &mut guard)?;
        self.run_cached_execution(cached, input)
    }

    pub fn detect_gray_with_timing(
        &self,
        image: ImageBuffer<image::Luma<u8>, Vec<u8>>,
    ) -> Result<(Vec<DetectedTag>, GpuTimingReport), ()> {
        self.validate_settings()?;

        let decimate_factor = self.settings.decimate.unwrap_or(1) as u32;
        let aligned_input = crop_image_to_multiple(image, 4 * decimate_factor)?;
        let input_gpu_image =
            crate::GPUImage::from_image_buffer(self.device.clone(), aligned_input);
        self.detect_gpu_buffer_with_timing_single_submit(
            input_gpu_image.image.descriptor(),
            input_gpu_image.size,
        )
    }

    fn validate_settings(&self) -> Result<(), ()> {
        if let Some(factor) = self.settings.decimate
            && !is_power_of_two(factor)
        {
            return Err(());
        }

        if self.settings.decode.cell_size == 0
            || self.settings.decode.cell_span == 0
            || self
                .settings
                .blob_pair_filter
                .max_cluster_pixels_perimeter_scale
                == 0
        {
            return Err(());
        }

        Ok(())
    }

    fn build_detected_tags(
        &self,
        fitted_quad_words: &[u32],
        fitted_quad_count: u32,
        bits_words: &[u32],
    ) -> Vec<DetectedTag> {
        let cells_per_tag = Self::MARKER_SIZE_WITH_BORDERS * Self::MARKER_SIZE_WITH_BORDERS;

        let mut tags = Vec::with_capacity(fitted_quad_count as usize);
        for i in 0..(fitted_quad_count as usize) {
            let base = i * Self::FITTED_QUAD_WORDS_PER_QUAD;
            if base + Self::FITTED_QUAD_WORDS_PER_QUAD > fitted_quad_words.len() {
                break;
            }

            let mut corners = [[0.0f32; 2]; 4];
            for (corner_idx, corner) in corners.iter_mut().enumerate() {
                let x_word =
                    fitted_quad_words[base + Self::FITTED_QUAD_CORNERS_START_WORD + corner_idx * 2];
                let y_word = fitted_quad_words
                    [base + Self::FITTED_QUAD_CORNERS_START_WORD + corner_idx * 2 + 1];
                *corner = [f32::from_bits(x_word), f32::from_bits(y_word)];
            }

            let bits_base = i * cells_per_tag;
            if bits_base + cells_per_tag > bits_words.len() {
                continue;
            }
            let quad_bits_words = &bits_words[bits_base..(bits_base + cells_per_tag)];
            let Some((id, rotation, bits_with_border, payload_bits)) =
                self.decode_quad_candidate_bits(quad_bits_words)
            else {
                continue;
            };

            tags.push(DetectedTag {
                quad_index: i as u32,
                id: Some(id),
                blob_index: fitted_quad_words[base + Self::FITTED_QUAD_BLOB_INDEX_WORD],
                reversed_border: fitted_quad_words[base + Self::FITTED_QUAD_REVERSED_BORDER_WORD]
                    != 0,
                score: f32::from_bits(fitted_quad_words[base + Self::FITTED_QUAD_SCORE_WORD]),
                corners: Self::roll_corners(corners, rotation as usize),
                bits_with_border,
                payload_bits,
            });
        }

        tags
    }

    fn decode_quad_candidate_bits(
        &self,
        quad_bits_words: &[u32],
    ) -> Option<(u32, u8, Vec<u8>, Vec<u8>)> {
        let mut candidate_bits = quad_bits_words
            .iter()
            .map(|&word| if word == 0 { 0u8 } else { 1u8 })
            .collect::<Vec<_>>();
        let mut border_errors = Self::get_border_errors(&candidate_bits);

        if self.settings.decode.detect_inverted_marker {
            let mut inverted = candidate_bits
                .iter()
                .map(|&bit| if bit == 0 { 1 } else { 0 })
                .collect::<Vec<u8>>();
            let inverted_border_errors = Self::get_border_errors(&inverted);
            if inverted_border_errors < border_errors {
                candidate_bits.clear();
                candidate_bits.append(&mut inverted);
                border_errors = inverted_border_errors;
            }
        }

        let max_border_errors = (Self::APRILTAG_MARKER_SIZE * Self::APRILTAG_MARKER_SIZE) as f32
            * self.settings.decode.max_erroneous_border_bits_rate;
        if border_errors > max_border_errors as usize {
            return None;
        }

        let payload_bits = Self::extract_payload_bits(&candidate_bits);
        let payload_code = Self::bits_to_code(&payload_bits);
        let (id, rotation) = Self::identify_apriltag_36h11(payload_code, self.settings.apriltag)?;
        Some((id, rotation, candidate_bits, payload_bits))
    }

    fn get_border_errors(bits_with_border: &[u8]) -> usize {
        let size_with_borders = Self::MARKER_SIZE_WITH_BORDERS;
        let border_size = Self::MARKER_BORDER_BITS;
        if bits_with_border.len() != size_with_borders * size_with_borders {
            return usize::MAX;
        }

        let mut total = 0usize;
        for y in 0..size_with_borders {
            for k in 0..border_size {
                total += usize::from(bits_with_border[y * size_with_borders + k] != 0);
                total += usize::from(
                    bits_with_border[y * size_with_borders + (size_with_borders - 1 - k)] != 0,
                );
            }
        }
        for x in border_size..(size_with_borders - border_size) {
            for k in 0..border_size {
                total += usize::from(bits_with_border[k * size_with_borders + x] != 0);
                total += usize::from(
                    bits_with_border[(size_with_borders - 1 - k) * size_with_borders + x] != 0,
                );
            }
        }
        total
    }

    fn extract_payload_bits(bits_with_border: &[u8]) -> Vec<u8> {
        let marker_size_with_borders = Self::MARKER_SIZE_WITH_BORDERS;
        let marker_border_bits = Self::MARKER_BORDER_BITS;
        let payload_bits_per_tag = Self::APRILTAG_MARKER_SIZE * Self::APRILTAG_MARKER_SIZE;
        let mut payload_bits = Vec::with_capacity(payload_bits_per_tag);
        for y in marker_border_bits..(marker_size_with_borders - marker_border_bits) {
            for x in marker_border_bits..(marker_size_with_borders - marker_border_bits) {
                payload_bits.push(bits_with_border[y * marker_size_with_borders + x]);
            }
        }
        payload_bits
    }

    fn bits_to_code(payload_bits: &[u8]) -> u64 {
        payload_bits
            .iter()
            .take(Self::APRILTAG_MARKER_SIZE * Self::APRILTAG_MARKER_SIZE)
            .enumerate()
            .fold(
                0u64,
                |acc, (idx, bit)| {
                    if *bit == 0 { acc } else { acc | (1u64 << idx) }
                },
            )
    }

    fn identify_apriltag_36h11(
        payload_code: u64,
        apriltag_settings: AprilTagSettings,
    ) -> Option<(u32, u8)> {
        let max_corrected_bits = (apriltag_settings.max_correction_bits as f32
            * apriltag_settings.error_correction_rate)
            .floor() as u32;

        if max_corrected_bits == 0 {
            return APRILTAG_36H11_EXACT_LOOKUP.get(&payload_code).copied();
        }

        let mut best_distance = u32::MAX;
        let mut best_id = 0u32;
        let mut best_rotation = 0u8;

        for entry in APRILTAG_36H11_ROTATED_CODES.iter().copied() {
            let distance = (payload_code ^ entry.code).count_ones();
            if distance < best_distance {
                best_distance = distance;
                best_id = entry.id;
                best_rotation = entry.rotation;
            }
        }

        (best_distance <= max_corrected_bits).then_some((best_id, best_rotation))
    }

    fn rotate_code_ccw(code: u64, marker_size: usize) -> u64 {
        let mut rotated = 0u64;
        for y in 0..marker_size {
            for x in 0..marker_size {
                let bit_index = y * marker_size + x;
                if ((code >> bit_index) & 1) == 0 {
                    continue;
                }
                let new_y = marker_size - 1 - x;
                let new_x = y;
                let new_index = new_y * marker_size + new_x;
                rotated |= 1u64 << new_index;
            }
        }
        rotated
    }

    fn roll_corners(corners: [[f32; 2]; 4], rotation: usize) -> [[f32; 2]; 4] {
        let mut corrected = [[0.0f32; 2]; 4];
        let r = rotation % 4;
        for (i, corner) in corners.into_iter().enumerate() {
            corrected[(i + r) % 4] = corner;
        }
        corrected
    }

    fn create_compute_pipeline(
        device: &ComputeDevice,
        module_bytes: &[u32],
    ) -> Arc<ComputePipeline> {
        Arc::new(device.create_compute_pipeline(module_bytes, None))
    }

    fn create_descriptor_set(
        &self,
        _compute_pipeline: &Arc<ComputePipeline>,
        writes: Vec<WriteDescriptorSet>,
    ) -> Arc<DescriptorSet> {
        Arc::new(DescriptorSet { writes })
    }

    fn dispatch_with_push_constants_recorded_timed<T: Pod + Copy>(
        &self,
        timings: &mut PipelineTimings,
        name: &str,
        commands: &mut CommandRecorder<'_>,
        compute_pipeline: &Arc<ComputePipeline>,
        bindings: &[(u32, DescriptorBuffer)],
        push_constants: T,
        dispatch: [u32; 3],
    ) {
        timings.dispatch_with_push_constants(
            commands,
            name,
            compute_pipeline.as_ref(),
            bindings,
            &push_constants,
            dispatch,
        );
    }

    fn dispatch_indirect_with_push_constants_recorded_timed<T: Pod + Copy>(
        &self,
        timings: &mut PipelineTimings,
        name: &str,
        commands: &mut CommandRecorder<'_>,
        compute_pipeline: &Arc<ComputePipeline>,
        bindings: &[(u32, DescriptorBuffer)],
        push_constants: T,
        indirect_buffer: &GpuBuffer<u32>,
        indirect_offset: vk::DeviceSize,
    ) {
        timings.dispatch_indirect_with_push_constants(
            commands,
            name,
            compute_pipeline.as_ref(),
            bindings,
            &push_constants,
            indirect_buffer,
            indirect_offset,
        );
    }

    fn dispatch_with_push_constants_timed<T: Pod + Copy>(
        &self,
        timings: &mut PipelineTimings,
        name: &str,
        compute_pipeline: Arc<ComputePipeline>,
        descriptor_set: Arc<DescriptorSet>,
        push_constants: T,
        dispatch: [u32; 3],
    ) {
        let bindings: Vec<(u32, DescriptorBuffer)> = descriptor_set
            .writes
            .iter()
            .map(|write| (write.binding, write.buffer))
            .collect();
        self.device.run_commands(|commands| {
            timings.dispatch_with_push_constants(
                commands,
                name,
                compute_pipeline.as_ref(),
                &bindings,
                &push_constants,
                dispatch,
            );
        });
    }

    fn control_word_offset(word_index: u32) -> vk::DeviceSize {
        (word_index as vk::DeviceSize) * (std::mem::size_of::<u32>() as vk::DeviceSize)
    }

    fn control_counter_descriptor(control: &GpuBuffer<u32>, word_index: u32) -> DescriptorBuffer {
        control.descriptor_range(
            Self::control_word_offset(word_index),
            std::mem::size_of::<u32>() as vk::DeviceSize,
        )
    }

    fn control_dispatch_offset(word_index: u32) -> vk::DeviceSize {
        Self::control_word_offset(word_index)
    }

    fn new_u8_storage_buffer(&self, len: usize) -> GpuBuffer<u8> {
        self.device.create_buffer(
            len,
            ash::vk::BufferUsageFlags::STORAGE_BUFFER
                | ash::vk::BufferUsageFlags::TRANSFER_SRC
                | ash::vk::BufferUsageFlags::TRANSFER_DST,
            BufferMemory::DeviceLocal,
        )
    }

    fn new_u32_control_buffer(&self, len: usize) -> GpuBuffer<u32> {
        self.device.create_buffer(
            len,
            ash::vk::BufferUsageFlags::STORAGE_BUFFER
                | ash::vk::BufferUsageFlags::TRANSFER_SRC
                | ash::vk::BufferUsageFlags::TRANSFER_DST
                | ash::vk::BufferUsageFlags::INDIRECT_BUFFER,
            BufferMemory::DeviceLocal,
        )
    }

    fn new_u32_storage_buffer(&self, len: usize) -> GpuBuffer<u32> {
        self.device.create_buffer(
            len,
            ash::vk::BufferUsageFlags::STORAGE_BUFFER
                | ash::vk::BufferUsageFlags::TRANSFER_SRC
                | ash::vk::BufferUsageFlags::TRANSFER_DST,
            BufferMemory::DeviceLocal,
        )
    }

    fn new_zeroed_u32_storage_buffer(&self, len: usize) -> GpuBuffer<u32> {
        let buf = self.device.create_buffer(
            len,
            ash::vk::BufferUsageFlags::STORAGE_BUFFER
                | ash::vk::BufferUsageFlags::TRANSFER_SRC
                | ash::vk::BufferUsageFlags::TRANSFER_DST,
            BufferMemory::HostSequentialWrite,
        );
        self.device.fill_buffer_u32(&buf, 0);
        buf
    }

    fn new_zeroed_u32_counter_buffer(&self) -> GpuBuffer<u32> {
        self.device.upload_buffer(
            &[0u32],
            ash::vk::BufferUsageFlags::STORAGE_BUFFER
                | ash::vk::BufferUsageFlags::TRANSFER_SRC
                | ash::vk::BufferUsageFlags::TRANSFER_DST,
            false,
        )
    }

    fn read_counter(&self, counter: &GpuBuffer<u32>) -> u32 {
        counter.read(1)[0]
    }

    fn download_u32_buffer(&self, source: &GpuBuffer<u32>, len: usize) -> Vec<u32> {
        let destination = self.device.create_buffer(
            len,
            ash::vk::BufferUsageFlags::TRANSFER_DST,
            BufferMemory::HostRandomAccess,
        );
        self.device
            .copy_buffer(source, &destination, destination.byte_size());
        destination.read(len)
    }

    fn radix_sort_blob_diff_points(
        &self,
        input: &GpuBuffer<u32>,
        valid_points: u32,
        timings: &mut PipelineTimings,
    ) -> GpuBuffer<u32> {
        // Existing blob extent construction expects points sorted by (rep1, rep0).
        self.radix_sort_records_lexicographic(
            input,
            valid_points,
            6,
            1,
            0,
            Self::RADIX_KEY_TRANSFORM_NONE,
            "radix-sort-blob-diff-points",
            timings,
        )
    }

    fn rewrite_selected_blob_points_with_theta(
        &self,
        sorted_points: &GpuBuffer<u32>,
        extents_in: &GpuBuffer<u32>,
        point_extent_indices: &GpuBuffer<u32>,
        filtered_extents: &GpuBuffer<u32>,
        selected_points_out: &GpuBuffer<u32>,
        extent_count: u32,
        valid_points: u32,
        timings: &mut PipelineTimings,
    ) {
        let compute_pipeline = self
            .pipelines
            .rewrite_selected_blob_points_with_theta
            .clone();
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![
                WriteDescriptorSet::buffer(0, sorted_points.clone()),
                WriteDescriptorSet::buffer(1, extents_in.clone()),
                WriteDescriptorSet::buffer(2, point_extent_indices.clone()),
                WriteDescriptorSet::buffer(3, filtered_extents.clone()),
                WriteDescriptorSet::buffer(4, selected_points_out.clone()),
            ],
        );
        self.dispatch_with_push_constants_timed(
            timings,
            "filter-rewrite-selected-blob-points-with-theta",
            compute_pipeline,
            descriptor_set,
            RewriteSelectedBlobPointsPushConstants {
                extent_count,
                valid_points,
            },
            [
                dispatch_groups_1d(valid_points, Self::ONE_D_LOCAL_SIZE_X),
                1,
                1,
            ],
        );
    }

    fn radix_sort_selected_blob_points(
        &self,
        input: &GpuBuffer<u32>,
        valid_points: u32,
        timings: &mut PipelineTimings,
    ) -> GpuBuffer<u32> {
        self.radix_sort_records_lexicographic(
            input,
            valid_points,
            4,
            0,
            1,
            Self::RADIX_KEY_TRANSFORM_NONE,
            "radix-sort-selected-blob-points",
            timings,
        )
    }

    fn rebuild_filtered_extent_starts_from_sorted_points(
        &self,
        filtered_extents: &GpuBuffer<u32>,
        sorted_selected_points: &GpuBuffer<u32>,
        extent_count: u32,
        point_count: u32,
        timings: &mut PipelineTimings,
    ) {
        if extent_count == 0 || point_count == 0 {
            return;
        }

        let compute_pipeline = self.pipelines.rebuild_filtered_extent_starts.clone();
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![
                WriteDescriptorSet::buffer(0, sorted_selected_points.clone()),
                WriteDescriptorSet::buffer(1, filtered_extents.clone()),
            ],
        );
        self.dispatch_with_push_constants_timed(
            timings,
            "filter-rebuild-filtered-extent-starts",
            compute_pipeline,
            descriptor_set,
            RebuildFilteredExtentStartsPushConstants {
                extent_count,
                point_count,
            },
            [
                dispatch_groups_1d(point_count, Self::ONE_D_LOCAL_SIZE_X),
                1,
                1,
            ],
        );
    }

    fn radix_sort_peaks(
        &self,
        input: &GpuBuffer<u32>,
        valid_peaks: u32,
        timings: &mut PipelineTimings,
    ) -> GpuBuffer<u32> {
        // Existing peak extent construction expects peaks sorted by (blob_index, float(error)).
        self.radix_sort_records_lexicographic(
            input,
            valid_peaks,
            3,
            0,
            1,
            Self::RADIX_KEY_TRANSFORM_F32_ASC,
            "radix-sort-peaks",
            timings,
        )
    }

    fn radix_sort_records_lexicographic(
        &self,
        input: &GpuBuffer<u32>,
        valid_records: u32,
        words_per_record: u32,
        primary_key_word: u32,
        secondary_key_word: u32,
        secondary_key_transform: u32,
        timing_prefix: &str,
        timings: &mut PipelineTimings,
    ) -> GpuBuffer<u32> {
        let output_words = usize::max(1, valid_records as usize * words_per_record as usize);
        let sorted_records = self.new_u32_storage_buffer(output_words);

        if valid_records == 0 {
            return sorted_records;
        }

        if valid_records == 1 {
            let bytes = (words_per_record as ash::vk::DeviceSize)
                * (std::mem::size_of::<u32>() as ash::vk::DeviceSize);
            self.device
                .copy_buffer_region(input, 0, &sorted_records, 0, bytes);
            return sorted_records;
        }

        let sort_keys = self.new_u32_storage_buffer(valid_records as usize);
        let sorted_indices = self.new_u32_storage_buffer(valid_records as usize);

        let sort_storage = self.sorter.create_key_value_storage_buffer(valid_records);
        let secondary_sort_query =
            timings.reserve_radix_sort_queries(&format!("{timing_prefix}::secondary-key-sort"));
        let primary_sort_query =
            timings.reserve_radix_sort_queries(&format!("{timing_prefix}::primary-key-sort"));

        self.device.run_commands(|commands| {
            self.radix_init_keys_and_indices_recorded(
                commands,
                input,
                &sort_keys,
                &sorted_indices,
                valid_records,
                words_per_record,
                secondary_key_word,
                secondary_key_transform,
                timings,
                &format!("{timing_prefix}::sort-key-init"),
            );
            commands.barrier_shader_write_to_shader_read();

            self.sorter.record_sort_key_value_with_query_pool(
                commands,
                valid_records,
                &sort_keys,
                0,
                &sorted_indices,
                0,
                &sort_storage,
                0,
                Some((&timings.query_pool, secondary_sort_query)),
            );
            commands.barrier_shader_write_to_shader_read();

            self.radix_update_keys_from_indices_recorded(
                commands,
                input,
                &sorted_indices,
                &sort_keys,
                valid_records,
                words_per_record,
                primary_key_word,
                Self::RADIX_KEY_TRANSFORM_NONE,
                timings,
                &format!("{timing_prefix}::sort-key-update"),
            );
            commands.barrier_shader_write_to_shader_read();

            self.sorter.record_sort_key_value_with_query_pool(
                commands,
                valid_records,
                &sort_keys,
                0,
                &sorted_indices,
                0,
                &sort_storage,
                0,
                Some((&timings.query_pool, primary_sort_query)),
            );
            commands.barrier_shader_write_to_shader_read();

            self.radix_gather_records_by_indices_recorded(
                commands,
                input,
                &sorted_indices,
                &sorted_records,
                valid_records,
                words_per_record,
                timings,
                &format!("{timing_prefix}::gather"),
            );
        });

        sorted_records
    }

    fn radix_init_keys_and_indices(
        &self,
        input_records: &GpuBuffer<u32>,
        keys_out: &GpuBuffer<u32>,
        indices_out: &GpuBuffer<u32>,
        valid_points: u32,
        words_per_record: u32,
        key_word_index: u32,
        key_transform: u32,
        timings: &mut PipelineTimings,
        timing_name: &str,
    ) {
        self.device.run_commands(|commands| {
            self.radix_init_keys_and_indices_recorded(
                commands,
                input_records,
                keys_out,
                indices_out,
                valid_points,
                words_per_record,
                key_word_index,
                key_transform,
                timings,
                timing_name,
            );
        });
    }

    fn radix_init_keys_and_indices_recorded(
        &self,
        commands: &mut CommandRecorder<'_>,
        input_records: &GpuBuffer<u32>,
        keys_out: &GpuBuffer<u32>,
        indices_out: &GpuBuffer<u32>,
        valid_points: u32,
        words_per_record: u32,
        key_word_index: u32,
        key_transform: u32,
        timings: &mut PipelineTimings,
        timing_name: &str,
    ) {
        let compute_pipeline = self.pipelines.radix_init_keys_indices.clone();
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![
                WriteDescriptorSet::buffer(0, input_records.clone()),
                WriteDescriptorSet::buffer(1, keys_out.clone()),
                WriteDescriptorSet::buffer(2, indices_out.clone()),
            ],
        );
        self.dispatch_with_push_constants_recorded_from_descriptor_set_timed(
            timings,
            timing_name,
            commands,
            compute_pipeline,
            descriptor_set,
            RadixExtractPushConstants {
                valid_points,
                words_per_record,
                key_word_index,
                key_transform,
            },
            [
                dispatch_groups_1d(valid_points, Self::ONE_D_LOCAL_SIZE_X),
                1,
                1,
            ],
        );
    }

    fn radix_update_keys_from_indices(
        &self,
        input_records: &GpuBuffer<u32>,
        sorted_indices: &GpuBuffer<u32>,
        keys_out: &GpuBuffer<u32>,
        valid_points: u32,
        words_per_record: u32,
        key_word_index: u32,
        key_transform: u32,
        timings: &mut PipelineTimings,
        timing_name: &str,
    ) {
        self.device.run_commands(|commands| {
            self.radix_update_keys_from_indices_recorded(
                commands,
                input_records,
                sorted_indices,
                keys_out,
                valid_points,
                words_per_record,
                key_word_index,
                key_transform,
                timings,
                timing_name,
            );
        });
    }

    fn radix_update_keys_from_indices_recorded(
        &self,
        commands: &mut CommandRecorder<'_>,
        input_records: &GpuBuffer<u32>,
        sorted_indices: &GpuBuffer<u32>,
        keys_out: &GpuBuffer<u32>,
        valid_points: u32,
        words_per_record: u32,
        key_word_index: u32,
        key_transform: u32,
        timings: &mut PipelineTimings,
        timing_name: &str,
    ) {
        let compute_pipeline = self.pipelines.radix_keys_from_indices.clone();
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![
                WriteDescriptorSet::buffer(0, input_records.clone()),
                WriteDescriptorSet::buffer(1, sorted_indices.clone()),
                WriteDescriptorSet::buffer(2, keys_out.clone()),
            ],
        );
        self.dispatch_with_push_constants_recorded_from_descriptor_set_timed(
            timings,
            timing_name,
            commands,
            compute_pipeline,
            descriptor_set,
            RadixExtractPushConstants {
                valid_points,
                words_per_record,
                key_word_index,
                key_transform,
            },
            [
                dispatch_groups_1d(valid_points, Self::ONE_D_LOCAL_SIZE_X),
                1,
                1,
            ],
        );
    }

    fn radix_gather_records_by_indices(
        &self,
        input_records: &GpuBuffer<u32>,
        sorted_indices: &GpuBuffer<u32>,
        output_records: &GpuBuffer<u32>,
        valid_points: u32,
        words_per_record: u32,
        timings: &mut PipelineTimings,
        timing_name: &str,
    ) {
        self.device.run_commands(|commands| {
            self.radix_gather_records_by_indices_recorded(
                commands,
                input_records,
                sorted_indices,
                output_records,
                valid_points,
                words_per_record,
                timings,
                timing_name,
            );
        });
    }

    fn radix_gather_records_by_indices_recorded(
        &self,
        commands: &mut CommandRecorder<'_>,
        input_records: &GpuBuffer<u32>,
        sorted_indices: &GpuBuffer<u32>,
        output_records: &GpuBuffer<u32>,
        valid_points: u32,
        words_per_record: u32,
        timings: &mut PipelineTimings,
        timing_name: &str,
    ) {
        let compute_pipeline = self.pipelines.radix_gather_by_indices.clone();
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![
                WriteDescriptorSet::buffer(0, input_records.clone()),
                WriteDescriptorSet::buffer(1, sorted_indices.clone()),
                WriteDescriptorSet::buffer(2, output_records.clone()),
            ],
        );
        self.dispatch_with_push_constants_recorded_from_descriptor_set_timed(
            timings,
            timing_name,
            commands,
            compute_pipeline,
            descriptor_set,
            RadixGatherPushConstants {
                valid_points,
                words_per_record,
            },
            [
                dispatch_groups_1d(valid_points, Self::ONE_D_LOCAL_SIZE_X),
                1,
                1,
            ],
        );
    }

    fn dispatch_with_push_constants_recorded_from_descriptor_set_timed<T: Pod + Copy>(
        &self,
        timings: &mut PipelineTimings,
        name: &str,
        commands: &mut CommandRecorder<'_>,
        compute_pipeline: Arc<ComputePipeline>,
        descriptor_set: Arc<DescriptorSet>,
        push_constants: T,
        dispatch: [u32; 3],
    ) {
        let bindings: Vec<(u32, DescriptorBuffer)> = descriptor_set
            .writes
            .iter()
            .map(|write| (write.binding, write.buffer))
            .collect();
        timings.dispatch_with_push_constants(
            commands,
            name,
            compute_pipeline.as_ref(),
            &bindings,
            &push_constants,
            dispatch,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{AprilTagSettings, DetectedTag, DetectionSettings, Detector};
    use crate::ComputeDevice;
    use std::path::PathBuf;

    #[test]
    fn identifies_apriltag_36h11_code_and_rotation() {
        let marker_id = 23usize;
        let code = super::APRILTAG_36H11_CODES[marker_id];
        let decoded = Detector::identify_apriltag_36h11(code, AprilTagSettings::default());
        assert_eq!(decoded, Some((marker_id as u32, 0)));

        let rotated = Detector::rotate_code_ccw(code, Detector::APRILTAG_MARKER_SIZE);
        let decoded_rotated =
            Detector::identify_apriltag_36h11(rotated, AprilTagSettings::default());
        assert_eq!(decoded_rotated, Some((marker_id as u32, 1)));
    }

    #[test]
    fn rolls_corners_like_numpy_roll() {
        let corners = [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]];
        let rotated_start = [corners[1], corners[2], corners[3], corners[0]];
        let corrected = Detector::roll_corners(rotated_start, 1);
        assert_eq!(corrected, corners);
    }

    fn maybe_detector() -> Option<Detector> {
        std::panic::catch_unwind(|| {
            let device = ComputeDevice::new_default();
            Detector::new(
                device,
                DetectionSettings {
                    decimate: Some(2),
                    ..DetectionSettings::default()
                },
            )
        })
        .ok()
    }

    fn fixture_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("input")
            .join("test.jpg")
    }

    fn tags_signature(tags: &[DetectedTag]) -> Vec<(Option<u32>, bool, Vec<u8>, [i32; 8])> {
        let mut signature = Vec::with_capacity(tags.len());
        for tag in tags {
            let mut corners = [0i32; 8];
            for corner_index in 0..4 {
                corners[corner_index * 2] = (tag.corners[corner_index][0] * 1000.0).round() as i32;
                corners[corner_index * 2 + 1] =
                    (tag.corners[corner_index][1] * 1000.0).round() as i32;
            }
            signature.push((
                tag.id,
                tag.reversed_border,
                tag.payload_bits.clone(),
                corners,
            ));
        }
        signature.sort_unstable();
        signature
    }

    #[test]
    fn detection_is_deterministic_for_fixture_image() {
        let Some(detector) = maybe_detector() else {
            return;
        };

        let image_path = fixture_path();
        if !image_path.exists() {
            return;
        }

        let image = image::open(image_path).expect("failed to open fixture image");
        let gray = image.to_luma8();

        let baseline = detector
            .detect_gray(gray.clone())
            .expect("detection failed for baseline run");
        let baseline_signature = tags_signature(&baseline);

        for _ in 0..4 {
            let tags = detector
                .detect_gray(gray.clone())
                .expect("detection failed on repeated run");
            let signature = tags_signature(&tags);
            assert_eq!(signature, baseline_signature);
        }
    }
}
