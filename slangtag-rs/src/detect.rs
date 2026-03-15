use crate::gpu::{BufferMemory, ComputePipeline, DescriptorBuffer, GpuBuffer};
use crate::sort::RadixSorter;
use crate::{ComputeDevice, GPUImage, Size, compute_shader_path, include_u32};
use bytemuck::{Pod, Zeroable};
use image::{DynamicImage, GrayImage, ImageBuffer};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

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

pub struct Detector {
    device: ComputeDevice,
    settings: DetectionSettings,
    pipelines: DetectionPipelines,
    sorter: RadixSorter,
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
    count_nonzero_blob_diff_points: Arc<ComputePipeline>,
    filter_nonzero_blob_diff_points: Arc<ComputePipeline>,
    radix_init_keys_indices: Arc<ComputePipeline>,
    radix_keys_from_indices: Arc<ComputePipeline>,
    radix_gather_by_indices: Arc<ComputePipeline>,
    build_blob_pair_extents: Arc<ComputePipeline>,
    filter_blob_pair_extents: Arc<ComputePipeline>,
    rewrite_selected_blob_points_with_theta: Arc<ComputePipeline>,
    build_line_fit_points: Arc<ComputePipeline>,
    fit_line_errors_and_peaks: Arc<ComputePipeline>,
    count_valid_peaks: Arc<ComputePipeline>,
    filter_valid_peaks: Arc<ComputePipeline>,
    build_peak_extents: Arc<ComputePipeline>,
    fit_quads: Arc<ComputePipeline>,
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
            count_nonzero_blob_diff_points: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!(
                    "select-count-nonzero-blob-diff-points"
                )),
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
            filter_blob_pair_extents: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!("filter-filter-blob-pair-extents")),
            ),
            rewrite_selected_blob_points_with_theta: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!(
                    "filter-rewrite-selected-blob-points-with-theta"
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
            count_valid_peaks: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!("select-count-valid-peaks")),
            ),
            filter_valid_peaks: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!("select-filter-valid-peaks")),
            ),
            build_peak_extents: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!("filter-build-peak-extents")),
            ),
            fit_quads: Detector::create_compute_pipeline(
                device,
                include_u32!(compute_shader_path!("filter-fit-quads")),
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
struct FilterBlobPairExtentsPushConstants {
    extent_count: u32,
    tag_width: u32,
    reversed_border: u32,
    normal_border: u32,
    min_cluster_pixels: u32,
    max_cluster_pixels: u32,
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct RewriteSelectedBlobPointsPushConstants {
    extent_count: u32,
    valid_points: u32,
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct BuildLineFitPointsPushConstants {
    decimated_size: Size,
    point_count: u32,
    decimate: u32,
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct FitLineErrorsAndPeaksPushConstants {
    extent_count: u32,
    point_count: u32,
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

    pub fn new(device: ComputeDevice, settings: DetectionSettings) -> Self {
        let pipelines = DetectionPipelines::new(&device);
        let sorter = RadixSorter::new(device.clone());
        Self {
            device,
            settings,
            pipelines,
            sorter,
        }
    }

    pub fn detect(&self, image: DynamicImage) -> Result<Vec<DetectedTag>, ()> {
        let gray_image = image.to_luma8();
        self.detect_gray(gray_image)
    }

    pub fn detect_gray(
        &self,
        image: ImageBuffer<image::Luma<u8>, Vec<u8>>,
    ) -> Result<Vec<DetectedTag>, ()> {
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

        let decimate_factor = self.settings.decimate.unwrap_or(1) as u32;
        let aligned_input = crop_image_to_multiple(image, 4 * decimate_factor)?;
        let input_gpu_image =
            crate::GPUImage::from_image_buffer_fast(self.device.clone(), aligned_input);

        let decimated_image = match self.settings.decimate {
            Some(factor) => {
                let new_img = self.decimate(&input_gpu_image, factor);
                new_img
            }
            None => input_gpu_image.clone(),
        };

        let (minmax_image, minmax_size) = self.minmax(&decimated_image);
        let filtered_minmax_image = self.filter_minmax(&minmax_image, minmax_size);
        let thresholded_image = self.threshold(
            &decimated_image,
            &filtered_minmax_image,
            minmax_size,
            self.settings.min_white_black_diff,
        );
        let labels = self.ccl_init(&thresholded_image);
        self.ccl_compression(&labels, thresholded_image.size);
        self.ccl_merge(&labels, thresholded_image.size);
        self.ccl_compression(&labels, thresholded_image.size);

        let union_markers_size =
            self.new_zeroed_u32_storage_buffer(thresholded_image.size.total_pixels());
        self.ccl_final_labeling(&labels, &union_markers_size, thresholded_image.size);

        let blob_diff_words_per_point = 6usize;
        let blob_diff_points_per_offset = (thresholded_image.size.width as usize - 2)
            * (thresholded_image.size.height as usize - 2);
        let blob_diff_total_points = (blob_diff_points_per_offset * 4) as u32;
        let blob_diff_out = self
            .new_u32_storage_buffer(blob_diff_total_points as usize * blob_diff_words_per_point);
        self.blob_diff(
            &thresholded_image,
            &labels,
            &union_markers_size,
            &blob_diff_out,
            self.settings.min_blob_size,
        );

        let blob_diff_count = self.new_zeroed_u32_counter_buffer();
        self.count_nonzero_blob_diff_points(
            &blob_diff_out,
            &blob_diff_count,
            blob_diff_total_points,
        );
        let blob_diff_compacted_size = self.read_counter(&blob_diff_count);

        let blob_diff_compacted = self.new_u32_storage_buffer(
            usize::max(1, blob_diff_compacted_size as usize) * blob_diff_words_per_point,
        );
        let blob_diff_filter_count = self.new_zeroed_u32_counter_buffer();
        self.filter_nonzero_blob_diff_points(
            &blob_diff_out,
            &blob_diff_compacted,
            &blob_diff_filter_count,
            blob_diff_total_points,
        );
        let blob_diff_filtered_size = self.read_counter(&blob_diff_filter_count);

        let blob_diff_sorted =
            self.radix_sort_blob_diff_points(&blob_diff_compacted, blob_diff_filtered_size);

        let blob_extent_words_per_extent = 11usize;
        let blob_extent = self.new_u32_storage_buffer(
            usize::max(1, blob_diff_filtered_size as usize) * blob_extent_words_per_extent,
        );
        let blob_extent_count = self.new_zeroed_u32_counter_buffer();
        self.build_blob_pair_extents(
            &blob_diff_sorted,
            &blob_extent,
            &blob_extent_count,
            blob_diff_filtered_size,
        );
        let blob_extent_count_value = self.read_counter(&blob_extent_count);

        let filtered_blob_extent = self.new_u32_storage_buffer(
            usize::max(1, blob_extent_count_value as usize) * blob_extent_words_per_extent,
        );
        let selected_blob_extent_count = self.new_zeroed_u32_counter_buffer();
        let selected_blob_point_count = self.new_zeroed_u32_counter_buffer();
        let blob_pair_filter = self.settings.blob_pair_filter;
        let min_tag_width = blob_pair_filter.min_tag_width;
        let max_cluster_pixels = blob_pair_filter.max_cluster_pixels.unwrap_or(
            blob_pair_filter.max_cluster_pixels_perimeter_scale
                * (thresholded_image.size.width + thresholded_image.size.height),
        );
        self.filter_blob_pair_extents(
            &blob_extent,
            &filtered_blob_extent,
            &selected_blob_extent_count,
            &selected_blob_point_count,
            blob_extent_count_value,
            blob_pair_filter.tag_width,
            blob_pair_filter.reversed_border,
            blob_pair_filter.normal_border,
            blob_pair_filter.min_cluster_pixels,
            max_cluster_pixels,
        );
        let _selected_blob_extent_count_value = self.read_counter(&selected_blob_extent_count);
        let selected_blob_point_count_value = self.read_counter(&selected_blob_point_count);

        let selected_blob_point_words_per_point = 4usize;
        let selected_blob_points = self.new_u32_storage_buffer(
            usize::max(1, selected_blob_point_count_value as usize)
                * selected_blob_point_words_per_point,
        );
        self.rewrite_selected_blob_points_with_theta(
            &blob_diff_sorted,
            &blob_extent,
            &filtered_blob_extent,
            &selected_blob_points,
            blob_extent_count_value,
            blob_diff_filtered_size,
        );

        let selected_blob_sorted_points = self.radix_sort_selected_blob_points(
            &selected_blob_points,
            selected_blob_point_count_value,
        );

        let line_fit_point_words_per_point = 10usize;
        let line_fit_points = self.new_u32_storage_buffer(
            usize::max(1, selected_blob_point_count_value as usize)
                * line_fit_point_words_per_point,
        );
        self.build_line_fit_points(
            &selected_blob_sorted_points,
            &decimated_image,
            &line_fit_points,
            selected_blob_point_count_value,
            decimate_factor,
        );

        let errs =
            self.new_u32_storage_buffer(usize::max(1, selected_blob_point_count_value as usize));
        let filtered_errs =
            self.new_u32_storage_buffer(usize::max(1, selected_blob_point_count_value as usize));
        let peak_words_per_peak = 3usize;
        let peaks = self.new_u32_storage_buffer(
            usize::max(1, selected_blob_point_count_value as usize) * peak_words_per_peak,
        );
        self.fit_line_errors_and_peaks(
            &line_fit_points,
            &filtered_blob_extent,
            &errs,
            &filtered_errs,
            &peaks,
            blob_extent_count_value,
            selected_blob_point_count_value,
        );

        let peak_count = self.new_zeroed_u32_counter_buffer();
        if selected_blob_point_count_value > 0 {
            self.count_valid_peaks(&peaks, &peak_count, selected_blob_point_count_value);
        }
        let peak_count_value = self.read_counter(&peak_count);

        let compacted_peaks = self
            .new_u32_storage_buffer(usize::max(1, peak_count_value as usize) * peak_words_per_peak);
        let compacted_peak_count = self.new_zeroed_u32_counter_buffer();
        if selected_blob_point_count_value > 0 {
            self.filter_valid_peaks(
                &peaks,
                &compacted_peaks,
                &compacted_peak_count,
                selected_blob_point_count_value,
            );
        }
        let compacted_peak_count_value = self.read_counter(&compacted_peak_count);

        let sorted_peaks = self.radix_sort_peaks(&compacted_peaks, compacted_peak_count_value);

        let peak_extent_words_per_extent = 3usize;
        let peak_extents = self.new_u32_storage_buffer(
            usize::max(1, compacted_peak_count_value as usize) * peak_extent_words_per_extent,
        );
        let peak_extent_count = self.new_zeroed_u32_counter_buffer();
        self.build_peak_extents(
            &sorted_peaks,
            &peak_extents,
            &peak_extent_count,
            compacted_peak_count_value,
        );
        let peak_extent_count_value = self.read_counter(&peak_extent_count);

        let fitted_quad_words_per_quad = 15usize;
        let fitted_quads = self.new_u32_storage_buffer(
            usize::max(1, peak_extent_count_value as usize) * fitted_quad_words_per_quad,
        );
        let fitted_quad_count = self.new_zeroed_u32_counter_buffer();
        self.fit_quads(
            &sorted_peaks,
            &peak_extents,
            &line_fit_points,
            &filtered_blob_extent,
            &fitted_quads,
            &fitted_quad_count,
            peak_extent_count_value,
            blob_extent_count_value,
            min_tag_width,
            decimate_factor as f32,
        );
        let fitted_quad_count_value = self.read_counter(&fitted_quad_count);
        let mut detected_tags = Vec::new();

        if fitted_quad_count_value > 0 {
            let fitted_quad_words = self.download_u32_buffer(
                &fitted_quads,
                fitted_quad_count_value as usize * fitted_quad_words_per_quad,
            );
            let mut corners_words = Vec::with_capacity(fitted_quad_count_value as usize * 8);
            for i in 0..(fitted_quad_count_value as usize) {
                let base = i * fitted_quad_words_per_quad;
                corners_words.extend_from_slice(&fitted_quad_words[(base + 3)..(base + 11)]);
            }
            let corners = self.upload_u32_storage_buffer(&corners_words);

            let quad_param_words_per_quad = 12usize;
            let quad_params = self.new_u32_storage_buffer(
                usize::max(1, fitted_quad_count_value as usize) * quad_param_words_per_quad,
            );
            self.prepare_decode_quads(
                &input_gpu_image,
                &corners,
                &quad_params,
                fitted_quad_count_value,
                Self::MARKER_SIZE_WITH_BORDERS as u32,
                self.settings.decode.cell_size,
                self.settings.decode.min_stddev_otsu,
            );

            let bits_count = fitted_quad_count_value as usize * 8usize * 8usize;
            let bits = self.new_u32_storage_buffer(usize::max(1, bits_count));
            self.extract_candidate_bits(
                &input_gpu_image,
                &quad_params,
                &bits,
                fitted_quad_count_value,
                Self::MARKER_SIZE_WITH_BORDERS as u32,
                self.settings.decode.cell_size,
                self.settings.decode.cell_margin_pixels,
                self.settings.decode.cell_span,
            );

            let bits_words = self.download_u32_buffer(&bits, bits_count);
            detected_tags =
                self.build_detected_tags(&fitted_quad_words, fitted_quad_count_value, &bits_words);
        }

        Ok(detected_tags)
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

    fn dispatch_with_push_constants<T: Pod + Copy>(
        &self,
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
        self.device.dispatch_with_push_constants(
            compute_pipeline.as_ref(),
            &bindings,
            &push_constants,
            dispatch,
        );
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

    fn upload_u32_storage_buffer(&self, data: &[u32]) -> GpuBuffer<u32> {
        self.device.upload_buffer(
            data,
            ash::vk::BufferUsageFlags::STORAGE_BUFFER
                | ash::vk::BufferUsageFlags::TRANSFER_SRC
                | ash::vk::BufferUsageFlags::TRANSFER_DST,
            false,
        )
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

    fn decimate(&self, image: &GPUImage<u8>, factor: u8) -> GPUImage<u8> {
        let compute_pipeline = self.pipelines.decimate.clone();

        let decimated_size = crate::Size::new(
            image.size.width / factor as u32,
            image.size.height / factor as u32,
        );
        let decimated_image_buffer = self.new_u8_storage_buffer(decimated_size.total_pixels());
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![
                WriteDescriptorSet::buffer(0, image.image.clone()),
                WriteDescriptorSet::buffer(1, decimated_image_buffer.clone()),
            ],
        );

        let push_constants = DecimatePushConstants {
            input_size: image.size,
            decimated_size,
        };

        self.dispatch_with_push_constants(
            compute_pipeline,
            descriptor_set,
            push_constants,
            [image.size.width, image.size.height, 1],
        );

        GPUImage::new(self.device.clone(), decimated_image_buffer, decimated_size)
    }

    fn minmax(&self, decimated_image: &GPUImage<u8>) -> (GpuBuffer<u8>, Size) {
        let compute_pipeline = self.pipelines.minmax.clone();

        let minmax_size = Size::new(
            decimated_image.size.width / 4,
            decimated_image.size.height / 4,
        );
        let minmax_image = self.new_u8_storage_buffer(minmax_size.total_pixels() * 2);
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![
                WriteDescriptorSet::buffer(0, decimated_image.image.clone()),
                WriteDescriptorSet::buffer(1, minmax_image.clone()),
            ],
        );

        let push_constants = MinmaxPushConstants {
            input_size: decimated_image.size,
            minmax_size,
        };
        self.dispatch_with_push_constants(
            compute_pipeline,
            descriptor_set,
            push_constants,
            [minmax_size.width, minmax_size.height, 1],
        );

        (minmax_image, minmax_size)
    }

    fn filter_minmax(&self, minmax_image: &GpuBuffer<u8>, minmax_size: Size) -> GpuBuffer<u8> {
        let compute_pipeline = self.pipelines.filter_minmax.clone();

        let filtered_image = self.new_u8_storage_buffer(minmax_size.total_pixels() * 2);
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![
                WriteDescriptorSet::buffer(0, minmax_image.clone()),
                WriteDescriptorSet::buffer(1, filtered_image.clone()),
            ],
        );

        let push_constants = FilterMinmaxPushConstants {
            minmax_size,
            filtered_size: minmax_size,
        };
        self.dispatch_with_push_constants(
            compute_pipeline,
            descriptor_set,
            push_constants,
            [minmax_size.width, minmax_size.height, 1],
        );

        filtered_image
    }

    fn threshold(
        &self,
        decimated_image: &GPUImage<u8>,
        filtered_minmax_image: &GpuBuffer<u8>,
        filtered_size: Size,
        min_white_black_diff: u8,
    ) -> GPUImage<u8> {
        let compute_pipeline = self.pipelines.threshold.clone();

        let thresholded_size = decimated_image.size;
        let thresholded_image = self.new_u8_storage_buffer(thresholded_size.total_pixels());
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![
                WriteDescriptorSet::buffer(0, decimated_image.image.clone()),
                WriteDescriptorSet::buffer(1, filtered_minmax_image.clone()),
                WriteDescriptorSet::buffer(2, thresholded_image.clone()),
            ],
        );

        let push_constants = ThresholdPushConstants {
            decimated_size: decimated_image.size,
            filtered_size,
            thresholded_size,
            min_white_black_diff: min_white_black_diff as u32,
        };
        self.dispatch_with_push_constants(
            compute_pipeline,
            descriptor_set,
            push_constants,
            [thresholded_size.width, thresholded_size.height, 1],
        );

        GPUImage::new(self.device.clone(), thresholded_image, thresholded_size)
    }

    fn ccl_init(&self, thresholded_image: &GPUImage<u8>) -> GpuBuffer<u32> {
        let compute_pipeline = self.pipelines.ccl_init.clone();

        let labels = self.new_u32_storage_buffer(thresholded_image.size.total_pixels());
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![
                WriteDescriptorSet::buffer(0, thresholded_image.image.clone()),
                WriteDescriptorSet::buffer(1, labels.clone()),
            ],
        );

        let push_constants = CclPushConstants {
            image_size: thresholded_image.size,
        };
        self.dispatch_with_push_constants(
            compute_pipeline,
            descriptor_set,
            push_constants,
            [
                thresholded_image.size.width / 2,
                thresholded_image.size.height / 2,
                1,
            ],
        );

        labels
    }

    fn ccl_compression(&self, labels: &GpuBuffer<u32>, image_size: Size) {
        let compute_pipeline = self.pipelines.ccl_compression.clone();

        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![WriteDescriptorSet::buffer(0, labels.clone())],
        );

        let push_constants = CclPushConstants { image_size };
        self.dispatch_with_push_constants(
            compute_pipeline,
            descriptor_set,
            push_constants,
            [image_size.width / 2, image_size.height / 2, 1],
        );
    }

    fn ccl_merge(&self, labels: &GpuBuffer<u32>, image_size: Size) {
        let compute_pipeline = self.pipelines.ccl_merge.clone();

        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![WriteDescriptorSet::buffer(0, labels.clone())],
        );

        let push_constants = CclPushConstants { image_size };
        self.dispatch_with_push_constants(
            compute_pipeline,
            descriptor_set,
            push_constants,
            [image_size.width / 2, image_size.height / 2, 1],
        );
    }

    fn ccl_final_labeling(
        &self,
        labels: &GpuBuffer<u32>,
        union_markers_size: &GpuBuffer<u32>,
        image_size: Size,
    ) {
        let compute_pipeline = self.pipelines.ccl_final_labeling.clone();

        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![
                WriteDescriptorSet::buffer(0, labels.clone()),
                WriteDescriptorSet::buffer(1, union_markers_size.clone()),
            ],
        );

        let push_constants = CclPushConstants { image_size };
        self.dispatch_with_push_constants(
            compute_pipeline,
            descriptor_set,
            push_constants,
            [image_size.width / 2, image_size.height / 2, 1],
        );
    }

    fn blob_diff(
        &self,
        thresholded_image: &GPUImage<u8>,
        labels: &GpuBuffer<u32>,
        union_markers_size: &GpuBuffer<u32>,
        result: &GpuBuffer<u32>,
        min_blob_size: u32,
    ) {
        let compute_pipeline = self.pipelines.blob_diff.clone();
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![
                WriteDescriptorSet::buffer(0, thresholded_image.image.clone()),
                WriteDescriptorSet::buffer(1, labels.clone()),
                WriteDescriptorSet::buffer(2, union_markers_size.clone()),
                WriteDescriptorSet::buffer(3, result.clone()),
            ],
        );

        self.dispatch_with_push_constants(
            compute_pipeline,
            descriptor_set,
            BlobDiffPushConstants {
                thresholded_size: thresholded_image.size,
                min_blob_size,
            },
            [
                thresholded_image.size.width,
                thresholded_image.size.height,
                1,
            ],
        );
    }

    fn count_nonzero_blob_diff_points(
        &self,
        input: &GpuBuffer<u32>,
        count_out: &GpuBuffer<u32>,
        total_points: u32,
    ) {
        let compute_pipeline = self.pipelines.count_nonzero_blob_diff_points.clone();
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![
                WriteDescriptorSet::buffer(0, input.clone()),
                WriteDescriptorSet::buffer(1, count_out.clone()),
            ],
        );
        self.dispatch_with_push_constants(
            compute_pipeline,
            descriptor_set,
            TotalPointsPushConstants { total_points },
            [
                dispatch_groups_1d(total_points, Self::ONE_D_LOCAL_SIZE_X),
                1,
                1,
            ],
        );
    }

    fn filter_nonzero_blob_diff_points(
        &self,
        input: &GpuBuffer<u32>,
        output: &GpuBuffer<u32>,
        output_count: &GpuBuffer<u32>,
        total_points: u32,
    ) {
        let compute_pipeline = self.pipelines.filter_nonzero_blob_diff_points.clone();
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![
                WriteDescriptorSet::buffer(0, input.clone()),
                WriteDescriptorSet::buffer(1, output.clone()),
                WriteDescriptorSet::buffer(2, output_count.clone()),
            ],
        );
        self.dispatch_with_push_constants(
            compute_pipeline,
            descriptor_set,
            TotalPointsPushConstants { total_points },
            [
                dispatch_groups_1d(total_points, Self::ONE_D_LOCAL_SIZE_X),
                1,
                1,
            ],
        );
    }

    fn radix_sort_blob_diff_points(
        &self,
        input: &GpuBuffer<u32>,
        valid_points: u32,
    ) -> GpuBuffer<u32> {
        // Existing blob extent construction expects points sorted by (rep1, rep0).
        self.radix_sort_records_lexicographic(
            input,
            valid_points,
            6,
            1,
            0,
            Self::RADIX_KEY_TRANSFORM_NONE,
        )
    }

    fn build_blob_pair_extents(
        &self,
        sorted_points: &GpuBuffer<u32>,
        extents_out: &GpuBuffer<u32>,
        extent_count_out: &GpuBuffer<u32>,
        valid_points: u32,
    ) {
        let compute_pipeline = self.pipelines.build_blob_pair_extents.clone();
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![
                WriteDescriptorSet::buffer(0, sorted_points.clone()),
                WriteDescriptorSet::buffer(1, extents_out.clone()),
                WriteDescriptorSet::buffer(2, extent_count_out.clone()),
            ],
        );
        self.dispatch_with_push_constants(
            compute_pipeline,
            descriptor_set,
            TotalPointsPushConstants {
                total_points: valid_points,
            },
            [1, 1, 1],
        );
    }

    fn filter_blob_pair_extents(
        &self,
        extents_in: &GpuBuffer<u32>,
        extents_out: &GpuBuffer<u32>,
        selected_extent_count_out: &GpuBuffer<u32>,
        selected_point_count_out: &GpuBuffer<u32>,
        extent_count: u32,
        tag_width: u32,
        reversed_border: u32,
        normal_border: u32,
        min_cluster_pixels: u32,
        max_cluster_pixels: u32,
    ) {
        let compute_pipeline = self.pipelines.filter_blob_pair_extents.clone();
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![
                WriteDescriptorSet::buffer(0, extents_in.clone()),
                WriteDescriptorSet::buffer(1, extents_out.clone()),
                WriteDescriptorSet::buffer(2, selected_extent_count_out.clone()),
                WriteDescriptorSet::buffer(3, selected_point_count_out.clone()),
            ],
        );
        self.dispatch_with_push_constants(
            compute_pipeline,
            descriptor_set,
            FilterBlobPairExtentsPushConstants {
                extent_count,
                tag_width,
                reversed_border,
                normal_border,
                min_cluster_pixels,
                max_cluster_pixels,
            },
            [1, 1, 1],
        );
    }

    fn rewrite_selected_blob_points_with_theta(
        &self,
        sorted_points: &GpuBuffer<u32>,
        extents_in: &GpuBuffer<u32>,
        filtered_extents: &GpuBuffer<u32>,
        selected_points_out: &GpuBuffer<u32>,
        extent_count: u32,
        valid_points: u32,
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
                WriteDescriptorSet::buffer(2, filtered_extents.clone()),
                WriteDescriptorSet::buffer(3, selected_points_out.clone()),
            ],
        );
        self.dispatch_with_push_constants(
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
    ) -> GpuBuffer<u32> {
        self.radix_sort_records_lexicographic(
            input,
            valid_points,
            4,
            0,
            1,
            Self::RADIX_KEY_TRANSFORM_NONE,
        )
    }

    fn build_line_fit_points(
        &self,
        sorted_selected_points: &GpuBuffer<u32>,
        decimated_image: &GPUImage<u8>,
        line_fit_points_out: &GpuBuffer<u32>,
        point_count: u32,
        decimate: u32,
    ) {
        let compute_pipeline = self.pipelines.build_line_fit_points.clone();
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![
                WriteDescriptorSet::buffer(0, sorted_selected_points.clone()),
                WriteDescriptorSet::buffer(1, decimated_image.image.clone()),
                WriteDescriptorSet::buffer(2, line_fit_points_out.clone()),
            ],
        );
        self.dispatch_with_push_constants(
            compute_pipeline,
            descriptor_set,
            BuildLineFitPointsPushConstants {
                decimated_size: decimated_image.size,
                point_count,
                decimate,
            },
            [1, 1, 1],
        );
    }

    fn fit_line_errors_and_peaks(
        &self,
        line_fit_points: &GpuBuffer<u32>,
        filtered_extents: &GpuBuffer<u32>,
        errs_out: &GpuBuffer<u32>,
        filtered_errs_out: &GpuBuffer<u32>,
        peaks_out: &GpuBuffer<u32>,
        extent_count: u32,
        point_count: u32,
    ) {
        let compute_pipeline = self.pipelines.fit_line_errors_and_peaks.clone();
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![
                WriteDescriptorSet::buffer(0, line_fit_points.clone()),
                WriteDescriptorSet::buffer(1, filtered_extents.clone()),
                WriteDescriptorSet::buffer(2, errs_out.clone()),
                WriteDescriptorSet::buffer(3, filtered_errs_out.clone()),
                WriteDescriptorSet::buffer(4, peaks_out.clone()),
            ],
        );
        self.dispatch_with_push_constants(
            compute_pipeline,
            descriptor_set,
            FitLineErrorsAndPeaksPushConstants {
                extent_count,
                point_count,
            },
            [1, 1, 1],
        );
    }

    fn count_valid_peaks(
        &self,
        input: &GpuBuffer<u32>,
        count_out: &GpuBuffer<u32>,
        total_peaks: u32,
    ) {
        let compute_pipeline = self.pipelines.count_valid_peaks.clone();
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![
                WriteDescriptorSet::buffer(0, input.clone()),
                WriteDescriptorSet::buffer(1, count_out.clone()),
            ],
        );
        self.dispatch_with_push_constants(
            compute_pipeline,
            descriptor_set,
            TotalPointsPushConstants {
                total_points: total_peaks,
            },
            [
                dispatch_groups_1d(total_peaks, Self::ONE_D_LOCAL_SIZE_X),
                1,
                1,
            ],
        );
    }

    fn filter_valid_peaks(
        &self,
        input: &GpuBuffer<u32>,
        output: &GpuBuffer<u32>,
        output_count: &GpuBuffer<u32>,
        total_peaks: u32,
    ) {
        let compute_pipeline = self.pipelines.filter_valid_peaks.clone();
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![
                WriteDescriptorSet::buffer(0, input.clone()),
                WriteDescriptorSet::buffer(1, output.clone()),
                WriteDescriptorSet::buffer(2, output_count.clone()),
            ],
        );
        self.dispatch_with_push_constants(
            compute_pipeline,
            descriptor_set,
            TotalPointsPushConstants {
                total_points: total_peaks,
            },
            [
                dispatch_groups_1d(total_peaks, Self::ONE_D_LOCAL_SIZE_X),
                1,
                1,
            ],
        );
    }

    fn radix_sort_peaks(&self, input: &GpuBuffer<u32>, valid_peaks: u32) -> GpuBuffer<u32> {
        // Existing peak extent construction expects peaks sorted by (blob_index, float(error)).
        self.radix_sort_records_lexicographic(
            input,
            valid_peaks,
            3,
            0,
            1,
            Self::RADIX_KEY_TRANSFORM_F32_ASC,
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

        self.radix_init_keys_and_indices(
            input,
            &sort_keys,
            &sorted_indices,
            valid_records,
            words_per_record,
            secondary_key_word,
            secondary_key_transform,
        );

        let sort_storage = self.sorter.create_key_value_storage_buffer(valid_records);
        self.sorter.cmd_sort_key_value(
            valid_records,
            &sort_keys,
            0,
            &sorted_indices,
            0,
            &sort_storage,
            0,
        );

        self.radix_update_keys_from_indices(
            input,
            &sorted_indices,
            &sort_keys,
            valid_records,
            words_per_record,
            primary_key_word,
            Self::RADIX_KEY_TRANSFORM_NONE,
        );
        self.sorter.cmd_sort_key_value(
            valid_records,
            &sort_keys,
            0,
            &sorted_indices,
            0,
            &sort_storage,
            0,
        );

        self.radix_gather_records_by_indices(
            input,
            &sorted_indices,
            &sorted_records,
            valid_records,
            words_per_record,
        );

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
        self.dispatch_with_push_constants(
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
        self.dispatch_with_push_constants(
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
        self.dispatch_with_push_constants(
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

    fn build_peak_extents(
        &self,
        sorted_peaks: &GpuBuffer<u32>,
        peak_extents_out: &GpuBuffer<u32>,
        peak_extent_count_out: &GpuBuffer<u32>,
        valid_peaks: u32,
    ) {
        let compute_pipeline = self.pipelines.build_peak_extents.clone();
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![
                WriteDescriptorSet::buffer(0, sorted_peaks.clone()),
                WriteDescriptorSet::buffer(1, peak_extents_out.clone()),
                WriteDescriptorSet::buffer(2, peak_extent_count_out.clone()),
            ],
        );
        self.dispatch_with_push_constants(
            compute_pipeline,
            descriptor_set,
            TotalPointsPushConstants {
                total_points: valid_peaks,
            },
            [1, 1, 1],
        );
    }

    fn fit_quads(
        &self,
        sorted_peaks: &GpuBuffer<u32>,
        peak_extents: &GpuBuffer<u32>,
        line_fit_points: &GpuBuffer<u32>,
        filtered_blob_extents: &GpuBuffer<u32>,
        fitted_quads_out: &GpuBuffer<u32>,
        fitted_quad_count_out: &GpuBuffer<u32>,
        peak_extent_count: u32,
        filtered_blob_extent_count: u32,
        min_tag_width: u32,
        quad_decimate: f32,
    ) {
        let compute_pipeline = self.pipelines.fit_quads.clone();
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![
                WriteDescriptorSet::buffer(0, sorted_peaks.clone()),
                WriteDescriptorSet::buffer(1, peak_extents.clone()),
                WriteDescriptorSet::buffer(2, line_fit_points.clone()),
                WriteDescriptorSet::buffer(3, filtered_blob_extents.clone()),
                WriteDescriptorSet::buffer(4, fitted_quads_out.clone()),
                WriteDescriptorSet::buffer(5, fitted_quad_count_out.clone()),
            ],
        );
        self.dispatch_with_push_constants(
            compute_pipeline,
            descriptor_set,
            FitQuadsPushConstants {
                peak_extent_count,
                filtered_blob_extent_count,
                max_nmaxima: self.settings.quad_fit.max_nmaxima,
                max_line_fit_mse: self.settings.quad_fit.max_line_fit_mse,
                cos_critical_rad: self.settings.quad_fit.cos_critical_rad,
                min_tag_width,
                quad_decimate,
            },
            [u32::max(1, peak_extent_count), 1, 1],
        );
    }

    fn prepare_decode_quads(
        &self,
        image_gray: &GPUImage<u8>,
        corners: &GpuBuffer<u32>,
        quad_params_out: &GpuBuffer<u32>,
        quad_count: u32,
        marker_size_with_borders: u32,
        cell_size: u32,
        min_stddev_otsu: f32,
    ) {
        let compute_pipeline = self.pipelines.prepare_decode_quads.clone();
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![
                WriteDescriptorSet::buffer(0, image_gray.image.clone()),
                WriteDescriptorSet::buffer(1, corners.clone()),
                WriteDescriptorSet::buffer(2, quad_params_out.clone()),
            ],
        );
        self.dispatch_with_push_constants(
            compute_pipeline,
            descriptor_set,
            PrepareDecodeQuadsPushConstants {
                image_size: image_gray.size,
                quad_count,
                marker_size_with_borders,
                cell_size,
                min_stddev_otsu,
            },
            [
                dispatch_groups_1d(quad_count, Self::ONE_D_LOCAL_SIZE_X),
                1,
                1,
            ],
        );
    }

    fn extract_candidate_bits(
        &self,
        image_gray: &GPUImage<u8>,
        quad_params: &GpuBuffer<u32>,
        bits_out: &GpuBuffer<u32>,
        quad_count: u32,
        marker_size_with_borders: u32,
        cell_size: u32,
        cell_margin_pixels: u32,
        cell_span: u32,
    ) {
        let compute_pipeline = self.pipelines.extract_candidate_bits.clone();
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![
                WriteDescriptorSet::buffer(0, image_gray.image.clone()),
                WriteDescriptorSet::buffer(1, quad_params.clone()),
                WriteDescriptorSet::buffer(2, bits_out.clone()),
            ],
        );
        let bits_count = quad_count * marker_size_with_borders * marker_size_with_borders;
        self.dispatch_with_push_constants(
            compute_pipeline,
            descriptor_set,
            ExtractCandidateBitsPushConstants {
                image_size: image_gray.size,
                quad_count,
                marker_size_with_borders,
                cell_size,
                cell_margin_pixels,
                cell_span,
            },
            [
                dispatch_groups_1d(bits_count, Self::ONE_D_LOCAL_SIZE_X),
                1,
                1,
            ],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{AprilTagSettings, Detector};

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
}
