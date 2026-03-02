use crate::{ComputeDevice, GPUImage, Size, compute_shader_path};
use bytemuck::{Pod, Zeroable};
use image::{DynamicImage, GrayImage};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use vulkano::buffer::BufferContents;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::sync::{self, GpuFuture};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    pipeline::{
        ComputePipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    shader::{ShaderModuleCreateInfo, spirv::bytes_to_words},
};

include!("apriltag36h11_codes.rs");

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
        by_code.entry(entry.code).or_insert((entry.id, entry.rotation));
    }
    by_code
});

pub struct DetectionSettings {
    pub decimate: Option<u8>,
    pub min_white_black_diff: u8,
    pub min_blob_size: u32,
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
    prepare_blob_diff_points: Arc<ComputePipeline>,
    bitonic_sort_blob_diff_points: Arc<ComputePipeline>,
    build_blob_pair_extents: Arc<ComputePipeline>,
    filter_blob_pair_extents: Arc<ComputePipeline>,
    rewrite_selected_blob_points_with_theta: Arc<ComputePipeline>,
    prepare_selected_blob_points: Arc<ComputePipeline>,
    bitonic_sort_selected_blob_points: Arc<ComputePipeline>,
    build_line_fit_points: Arc<ComputePipeline>,
    fit_line_errors_and_peaks: Arc<ComputePipeline>,
    count_valid_peaks: Arc<ComputePipeline>,
    filter_valid_peaks: Arc<ComputePipeline>,
    prepare_peaks: Arc<ComputePipeline>,
    bitonic_sort_peaks: Arc<ComputePipeline>,
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
                include_bytes!(compute_shader_path!("threshold-decimate")),
            ),
            minmax: Detector::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!("threshold-minmax")),
            ),
            filter_minmax: Detector::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!("threshold-filter-minmax")),
            ),
            threshold: Detector::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!("threshold-threshold")),
            ),
            ccl_init: Detector::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!("ccl-init")),
            ),
            ccl_compression: Detector::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!("ccl-compression")),
            ),
            ccl_merge: Detector::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!("ccl-merge")),
            ),
            ccl_final_labeling: Detector::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!("ccl-final-labeling")),
            ),
            blob_diff: Detector::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!("blob-blob-diff")),
            ),
            count_nonzero_blob_diff_points: Detector::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!(
                    "select-count-nonzero-blob-diff-points"
                )),
            ),
            filter_nonzero_blob_diff_points: Detector::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!(
                    "select-filter-nonzero-blob-diff-points"
                )),
            ),
            prepare_blob_diff_points: Detector::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!("sort-prepare-blob-diff-points")),
            ),
            bitonic_sort_blob_diff_points: Detector::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!(
                    "sort-bitonic-sort-blob-diff-points"
                )),
            ),
            build_blob_pair_extents: Detector::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!("filter-build-blob-pair-extents")),
            ),
            filter_blob_pair_extents: Detector::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!("filter-filter-blob-pair-extents")),
            ),
            rewrite_selected_blob_points_with_theta: Detector::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!(
                    "filter-rewrite-selected-blob-points-with-theta"
                )),
            ),
            prepare_selected_blob_points: Detector::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!("sort-prepare-selected-blob-points")),
            ),
            bitonic_sort_selected_blob_points: Detector::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!(
                    "sort-bitonic-sort-selected-blob-points"
                )),
            ),
            build_line_fit_points: Detector::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!("filter-build-line-fit-points")),
            ),
            fit_line_errors_and_peaks: Detector::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!("filter-fit-line-errors-and-peaks")),
            ),
            count_valid_peaks: Detector::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!("select-count-valid-peaks")),
            ),
            filter_valid_peaks: Detector::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!("select-filter-valid-peaks")),
            ),
            prepare_peaks: Detector::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!("sort-prepare-peaks")),
            ),
            bitonic_sort_peaks: Detector::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!("sort-bitonic-sort-peaks")),
            ),
            build_peak_extents: Detector::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!("filter-build-peak-extents")),
            ),
            fit_quads: Detector::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!("filter-fit-quads")),
            ),
            prepare_decode_quads: Detector::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!("decode-prepare-decode-quads")),
            ),
            extract_candidate_bits: Detector::create_compute_pipeline(
                device,
                include_bytes!(compute_shader_path!("decode-extract-candidate-bits")),
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
        }
    }
}

fn is_power_of_two(n: u8) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

fn next_power_of_two(value: u32) -> u32 {
    if value <= 1 {
        1
    } else {
        value.next_power_of_two()
    }
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
struct SortPreparePushConstants {
    valid_points: u32,
    total_points: u32,
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct BitonicSortPushConstants {
    total_points: u32,
    j: u32,
    k: u32,
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
    const APRILTAG_ERROR_CORRECTION_RATE: f32 = 0.6;
    const APRILTAG_MAX_CORRECTION_BITS: u32 = 0;
    const MAX_ERRONEOUS_BITS_IN_BORDER_RATE: f32 = 0.35;
    const DETECT_INVERTED_MARKER: bool = true;

    pub fn new(device: ComputeDevice, settings: DetectionSettings) -> Self {
        let pipelines = DetectionPipelines::new(&device);
        Self {
            device,
            settings,
            pipelines,
        }
    }

    pub fn detect(&self, image: DynamicImage) -> Result<Vec<DetectedTag>, ()> {
        if let Some(factor) = self.settings.decimate
            && !is_power_of_two(factor)
        {
            return Err(());
        }

        let decimate_factor = self.settings.decimate.unwrap_or(1) as u32;
        let aligned_input = crop_image_to_multiple(image.into_luma8(), 4 * decimate_factor)?;
        let input_gpu_image =
            crate::GPUImage::from_image_buffer(self.device.clone(), aligned_input);

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

        let blob_diff_sort_points = next_power_of_two(blob_diff_filtered_size);
        let blob_diff_sorted = self.new_u32_storage_buffer(
            usize::max(1, blob_diff_sort_points as usize) * blob_diff_words_per_point,
        );
        self.prepare_blob_diff_points(
            &blob_diff_compacted,
            &blob_diff_sorted,
            blob_diff_filtered_size,
            blob_diff_sort_points,
        );
        self.bitonic_sort_blob_diff_points(&blob_diff_sorted, blob_diff_sort_points);

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
        let min_tag_width = 3;
        self.filter_blob_pair_extents(
            &blob_extent,
            &filtered_blob_extent,
            &selected_blob_extent_count,
            &selected_blob_point_count,
            blob_extent_count_value,
            min_tag_width,
            1,
            1,
            24,
            4 * (thresholded_image.size.width + thresholded_image.size.height),
        );
        let selected_blob_extent_count_value = self.read_counter(&selected_blob_extent_count);
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

        let selected_blob_sort_points = next_power_of_two(selected_blob_point_count_value);
        let selected_blob_sorted_points = self.new_u32_storage_buffer(
            usize::max(1, selected_blob_sort_points as usize) * selected_blob_point_words_per_point,
        );
        self.prepare_selected_blob_points(
            &selected_blob_points,
            &selected_blob_sorted_points,
            selected_blob_point_count_value,
            selected_blob_sort_points,
        );
        self.bitonic_sort_selected_blob_points(
            &selected_blob_sorted_points,
            selected_blob_sort_points,
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

        let peak_sort_points = next_power_of_two(compacted_peak_count_value);
        let sorted_peaks = self
            .new_u32_storage_buffer(usize::max(1, peak_sort_points as usize) * peak_words_per_peak);
        self.prepare_peaks(
            &compacted_peaks,
            &sorted_peaks,
            compacted_peak_count_value,
            peak_sort_points,
        );
        self.bitonic_sort_peaks(&sorted_peaks, peak_sort_points);

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
                8,
                4,
                5.0,
            );

            let bits_count = fitted_quad_count_value as usize * 8usize * 8usize;
            let bits = self.new_u32_storage_buffer(usize::max(1, bits_count));
            self.extract_candidate_bits(
                &input_gpu_image,
                &quad_params,
                &bits,
                fitted_quad_count_value,
                8,
                4,
                0,
                4,
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

        if Self::DETECT_INVERTED_MARKER {
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
            * Self::MAX_ERRONEOUS_BITS_IN_BORDER_RATE;
        if border_errors > max_border_errors as usize {
            return None;
        }

        let payload_bits = Self::extract_payload_bits(&candidate_bits);
        let payload_code = Self::bits_to_code(&payload_bits);
        let (id, rotation) = Self::identify_apriltag_36h11(payload_code)?;
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

    fn identify_apriltag_36h11(payload_code: u64) -> Option<(u32, u8)> {
        let max_corrected_bits = (Self::APRILTAG_MAX_CORRECTION_BITS as f32
            * Self::APRILTAG_ERROR_CORRECTION_RATE)
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
        module_bytes: &[u8],
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

        let stage = PipelineShaderStageCreateInfo::new(entry_point);
        let layout = PipelineLayout::new(
            device.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.device.clone())
                .unwrap(),
        )
        .expect("failed to create pipeline layout");

        ComputePipeline::new(
            device.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .expect("failed to create compute pipeline")
    }

    fn create_descriptor_set(
        &self,
        compute_pipeline: &Arc<ComputePipeline>,
        writes: Vec<WriteDescriptorSet>,
    ) -> Arc<DescriptorSet> {
        let pipeline_layout = compute_pipeline.layout();
        let descriptor_set_layout = pipeline_layout
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

    fn dispatch_with_push_constants<T: Pod + Copy + BufferContents>(
        &self,
        compute_pipeline: Arc<ComputePipeline>,
        descriptor_set: Arc<DescriptorSet>,
        push_constants: T,
        dispatch: [u32; 3],
    ) {
        let mut builder = AutoCommandBufferBuilder::primary(
            self.device.command_buffer_allocator.clone(),
            self.device.queue_family_index,
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("failed to create command buffer builder");

        unsafe {
            builder
                .bind_pipeline_compute(compute_pipeline.clone())
                .expect("failed to bind compute pipeline")
                .bind_descriptor_sets(
                    vulkano::pipeline::PipelineBindPoint::Compute,
                    compute_pipeline.layout().clone(),
                    0,
                    descriptor_set,
                )
                .expect("failed to bind descriptor set")
                .push_constants(compute_pipeline.layout().clone(), 0, push_constants)
                .expect("failed to push constants")
                .dispatch(dispatch)
                .expect("failed to dispatch compute shader");
        }

        let command_buffer = builder.build().expect("failed to build command buffer");
        let future = sync::now(self.device.device.clone())
            .then_execute(self.device.queue.clone(), command_buffer)
            .expect("failed to execute command buffer")
            .then_signal_fence_and_flush()
            .expect("failed to flush command buffer");
        future.wait(None).expect("failed to wait for compute work");
    }

    fn new_u8_storage_buffer(&self, len: usize) -> Subbuffer<[u8]> {
        Buffer::new_unsized(
            self.device.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            len.try_into().expect("buffer length exceeds GPU limits"),
        )
        .expect("failed to create u8 storage buffer")
    }

    fn new_u32_storage_buffer(&self, len: usize) -> Subbuffer<[u32]> {
        Buffer::new_unsized(
            self.device.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            len.try_into().expect("buffer length exceeds GPU limits"),
        )
        .expect("failed to create u32 storage buffer")
    }

    fn new_zeroed_u32_storage_buffer(&self, len: usize) -> Subbuffer<[u32]> {
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
            std::iter::repeat(0u32).take(len),
        )
        .expect("failed to create zeroed u32 storage buffer")
    }

    fn new_zeroed_u32_counter_buffer(&self) -> Subbuffer<[u32]> {
        Buffer::from_iter(
            self.device.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            std::iter::once(0u32),
        )
        .expect("failed to create zeroed u32 counter buffer")
    }

    fn read_counter(&self, counter: &Subbuffer<[u32]>) -> u32 {
        counter.read().expect("failed to map counter buffer")[0]
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
            len.try_into().expect("buffer length exceeds GPU limits"),
        )
        .expect("failed to create destination buffer");

        let mut builder = AutoCommandBufferBuilder::primary(
            self.device.command_buffer_allocator.clone(),
            self.device.queue_family_index,
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("failed to create command buffer builder");

        builder
            .copy_buffer(CopyBufferInfo::buffers(source.clone(), destination.clone()))
            .expect("failed to copy u32 buffer to host");

        let command_buffer = builder.build().expect("failed to build command buffer");
        let future = sync::now(self.device.device.clone())
            .then_execute(self.device.queue.clone(), command_buffer)
            .expect("failed to execute command buffer")
            .then_signal_fence_and_flush()
            .expect("failed to flush command buffer");
        future.wait(None).expect("failed to wait for buffer copy");

        destination
            .read()
            .expect("failed to map destination buffer")
            .to_vec()
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

    fn minmax(&self, decimated_image: &GPUImage<u8>) -> (Subbuffer<[u8]>, Size) {
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

    fn filter_minmax(&self, minmax_image: &Subbuffer<[u8]>, minmax_size: Size) -> Subbuffer<[u8]> {
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
        filtered_minmax_image: &Subbuffer<[u8]>,
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

    fn ccl_init(&self, thresholded_image: &GPUImage<u8>) -> Subbuffer<[u32]> {
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

    fn ccl_compression(&self, labels: &Subbuffer<[u32]>, image_size: Size) {
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

    fn ccl_merge(&self, labels: &Subbuffer<[u32]>, image_size: Size) {
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
        labels: &Subbuffer<[u32]>,
        union_markers_size: &Subbuffer<[u32]>,
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
        labels: &Subbuffer<[u32]>,
        union_markers_size: &Subbuffer<[u32]>,
        result: &Subbuffer<[u32]>,
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
        input: &Subbuffer<[u32]>,
        count_out: &Subbuffer<[u32]>,
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
        input: &Subbuffer<[u32]>,
        output: &Subbuffer<[u32]>,
        output_count: &Subbuffer<[u32]>,
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

    fn prepare_blob_diff_points(
        &self,
        input: &Subbuffer<[u32]>,
        output: &Subbuffer<[u32]>,
        valid_points: u32,
        total_points: u32,
    ) {
        let compute_pipeline = self.pipelines.prepare_blob_diff_points.clone();
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![
                WriteDescriptorSet::buffer(0, input.clone()),
                WriteDescriptorSet::buffer(1, output.clone()),
            ],
        );
        self.dispatch_with_push_constants(
            compute_pipeline,
            descriptor_set,
            SortPreparePushConstants {
                valid_points,
                total_points,
            },
            [
                dispatch_groups_1d(total_points, Self::ONE_D_LOCAL_SIZE_X),
                1,
                1,
            ],
        );
    }

    fn bitonic_sort_blob_diff_points(&self, points: &Subbuffer<[u32]>, total_points: u32) {
        if total_points <= 1 {
            return;
        }

        let compute_pipeline = self.pipelines.bitonic_sort_blob_diff_points.clone();
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![WriteDescriptorSet::buffer(0, points.clone())],
        );

        let mut k = 2;
        while k <= total_points {
            let mut j = k / 2;
            while j > 0 {
                self.dispatch_with_push_constants(
                    compute_pipeline.clone(),
                    descriptor_set.clone(),
                    BitonicSortPushConstants { total_points, j, k },
                    [
                        dispatch_groups_1d(total_points, Self::ONE_D_LOCAL_SIZE_X),
                        1,
                        1,
                    ],
                );
                j /= 2;
            }
            k *= 2;
        }
    }

    fn build_blob_pair_extents(
        &self,
        sorted_points: &Subbuffer<[u32]>,
        extents_out: &Subbuffer<[u32]>,
        extent_count_out: &Subbuffer<[u32]>,
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
        extents_in: &Subbuffer<[u32]>,
        extents_out: &Subbuffer<[u32]>,
        selected_extent_count_out: &Subbuffer<[u32]>,
        selected_point_count_out: &Subbuffer<[u32]>,
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
        sorted_points: &Subbuffer<[u32]>,
        extents_in: &Subbuffer<[u32]>,
        filtered_extents: &Subbuffer<[u32]>,
        selected_points_out: &Subbuffer<[u32]>,
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

    fn prepare_selected_blob_points(
        &self,
        input: &Subbuffer<[u32]>,
        output: &Subbuffer<[u32]>,
        valid_points: u32,
        total_points: u32,
    ) {
        let compute_pipeline = self.pipelines.prepare_selected_blob_points.clone();
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![
                WriteDescriptorSet::buffer(0, input.clone()),
                WriteDescriptorSet::buffer(1, output.clone()),
            ],
        );
        self.dispatch_with_push_constants(
            compute_pipeline,
            descriptor_set,
            SortPreparePushConstants {
                valid_points,
                total_points,
            },
            [
                dispatch_groups_1d(total_points, Self::ONE_D_LOCAL_SIZE_X),
                1,
                1,
            ],
        );
    }

    fn bitonic_sort_selected_blob_points(&self, points: &Subbuffer<[u32]>, total_points: u32) {
        if total_points <= 1 {
            return;
        }

        let compute_pipeline = self.pipelines.bitonic_sort_selected_blob_points.clone();
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![WriteDescriptorSet::buffer(0, points.clone())],
        );

        let mut k = 2;
        while k <= total_points {
            let mut j = k / 2;
            while j > 0 {
                self.dispatch_with_push_constants(
                    compute_pipeline.clone(),
                    descriptor_set.clone(),
                    BitonicSortPushConstants { total_points, j, k },
                    [
                        dispatch_groups_1d(total_points, Self::ONE_D_LOCAL_SIZE_X),
                        1,
                        1,
                    ],
                );
                j /= 2;
            }
            k *= 2;
        }
    }

    fn build_line_fit_points(
        &self,
        sorted_selected_points: &Subbuffer<[u32]>,
        decimated_image: &GPUImage<u8>,
        line_fit_points_out: &Subbuffer<[u32]>,
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
        line_fit_points: &Subbuffer<[u32]>,
        filtered_extents: &Subbuffer<[u32]>,
        errs_out: &Subbuffer<[u32]>,
        filtered_errs_out: &Subbuffer<[u32]>,
        peaks_out: &Subbuffer<[u32]>,
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
        input: &Subbuffer<[u32]>,
        count_out: &Subbuffer<[u32]>,
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
        input: &Subbuffer<[u32]>,
        output: &Subbuffer<[u32]>,
        output_count: &Subbuffer<[u32]>,
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

    fn prepare_peaks(
        &self,
        input: &Subbuffer<[u32]>,
        output: &Subbuffer<[u32]>,
        valid_peaks: u32,
        total_peaks: u32,
    ) {
        let compute_pipeline = self.pipelines.prepare_peaks.clone();
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![
                WriteDescriptorSet::buffer(0, input.clone()),
                WriteDescriptorSet::buffer(1, output.clone()),
            ],
        );
        self.dispatch_with_push_constants(
            compute_pipeline,
            descriptor_set,
            SortPreparePushConstants {
                valid_points: valid_peaks,
                total_points: total_peaks,
            },
            [
                dispatch_groups_1d(total_peaks, Self::ONE_D_LOCAL_SIZE_X),
                1,
                1,
            ],
        );
    }

    fn bitonic_sort_peaks(&self, peaks: &Subbuffer<[u32]>, total_peaks: u32) {
        if total_peaks <= 1 {
            return;
        }

        let compute_pipeline = self.pipelines.bitonic_sort_peaks.clone();
        let descriptor_set = self.create_descriptor_set(
            &compute_pipeline,
            vec![WriteDescriptorSet::buffer(0, peaks.clone())],
        );

        let mut k = 2;
        while k <= total_peaks {
            let mut j = k / 2;
            while j > 0 {
                self.dispatch_with_push_constants(
                    compute_pipeline.clone(),
                    descriptor_set.clone(),
                    BitonicSortPushConstants {
                        total_points: total_peaks,
                        j,
                        k,
                    },
                    [
                        dispatch_groups_1d(total_peaks, Self::ONE_D_LOCAL_SIZE_X),
                        1,
                        1,
                    ],
                );
                j /= 2;
            }
            k *= 2;
        }
    }

    fn build_peak_extents(
        &self,
        sorted_peaks: &Subbuffer<[u32]>,
        peak_extents_out: &Subbuffer<[u32]>,
        peak_extent_count_out: &Subbuffer<[u32]>,
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
        sorted_peaks: &Subbuffer<[u32]>,
        peak_extents: &Subbuffer<[u32]>,
        line_fit_points: &Subbuffer<[u32]>,
        filtered_blob_extents: &Subbuffer<[u32]>,
        fitted_quads_out: &Subbuffer<[u32]>,
        fitted_quad_count_out: &Subbuffer<[u32]>,
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
                max_nmaxima: 10,
                max_line_fit_mse: 10.0,
                cos_critical_rad: 0.984_807_73,
                min_tag_width,
                quad_decimate,
            },
            [u32::max(1, peak_extent_count), 1, 1],
        );
    }

    fn prepare_decode_quads(
        &self,
        image_gray: &GPUImage<u8>,
        corners: &Subbuffer<[u32]>,
        quad_params_out: &Subbuffer<[u32]>,
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
        quad_params: &Subbuffer<[u32]>,
        bits_out: &Subbuffer<[u32]>,
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
    use super::Detector;

    #[test]
    fn identifies_apriltag_36h11_code_and_rotation() {
        let marker_id = 23usize;
        let code = super::APRILTAG_36H11_CODES[marker_id];
        let decoded = Detector::identify_apriltag_36h11(code);
        assert_eq!(decoded, Some((marker_id as u32, 0)));

        let rotated = Detector::rotate_code_ccw(code, Detector::APRILTAG_MARKER_SIZE);
        let decoded_rotated = Detector::identify_apriltag_36h11(rotated);
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
