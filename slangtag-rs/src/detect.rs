use crate::{ComputeDevice, GPUImage, Size, compute_shader_path};
use bytemuck::{Pod, Zeroable};
use image::{DynamicImage, GrayImage};
use std::sync::Arc;
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

fn canonicalize_labels_cpu(labels: &[u32]) -> Vec<u32> {
    let mut parents = labels.to_vec();
    let n = parents.len();
    for i in 0..n {
        let mut x = parents[i] as usize;
        let mut guard = 0usize;
        while x < n && parents[x] as usize != x && guard < n {
            x = parents[x] as usize;
            guard += 1;
        }
        let root = if x < n { x as u32 } else { parents[i] };
        let mut y = i;
        guard = 0;
        while (parents[y] as usize) < n && parents[y] != root && guard < n {
            let next = parents[y] as usize;
            parents[y] = root;
            y = next;
            guard += 1;
        }
        parents[i] = root;
    }
    parents
}

fn ccl_from_thresholded_cpu(thresholded: &[u8], size: Size) -> (Vec<u32>, Vec<u32>) {
    let width = size.width as usize;
    let height = size.height as usize;
    let n = width * height;
    let mut parents: Vec<usize> = (0..n).collect();

    fn find(parents: &mut [usize], mut x: usize) -> usize {
        while parents[x] != x {
            let p = parents[x];
            parents[x] = parents[p];
            x = parents[x];
        }
        x
    }

    fn unite(parents: &mut [usize], a: usize, b: usize) {
        let ra = find(parents, a);
        let rb = find(parents, b);
        if ra == rb {
            return;
        }
        if ra < rb {
            parents[rb] = ra;
        } else {
            parents[ra] = rb;
        }
    }

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let v = thresholded[idx];
            if v == 127 {
                continue;
            }
            if x > 0 {
                let nidx = idx - 1;
                if thresholded[nidx] == v {
                    unite(&mut parents, idx, nidx);
                }
            }
            if y > 0 {
                let nidx = idx - width;
                if thresholded[nidx] == v {
                    unite(&mut parents, idx, nidx);
                }
                if x > 0 {
                    let nidx = idx - width - 1;
                    if thresholded[nidx] == v {
                        unite(&mut parents, idx, nidx);
                    }
                }
                if x + 1 < width {
                    let nidx = idx - width + 1;
                    if thresholded[nidx] == v {
                        unite(&mut parents, idx, nidx);
                    }
                }
            }
        }
    }

    let mut labels = vec![0u32; n];
    for i in 0..n {
        if thresholded[i] == 127 {
            labels[i] = i as u32;
        } else {
            labels[i] = find(&mut parents, i) as u32;
        }
    }

    let mut union_sizes = vec![0u32; n];
    for i in 0..n {
        if thresholded[i] == 0 || thresholded[i] == 255 {
            let root = labels[i] as usize;
            if root < n {
                union_sizes[root] = union_sizes[root].saturating_add(1);
            }
        }
    }

    (labels, union_sizes)
}

fn ccl_from_thresholded_cpu_dual_connectivity(
    thresholded: &[u8],
    size: Size,
) -> (Vec<u32>, Vec<u32>) {
    let width = size.width as usize;
    let height = size.height as usize;
    let n = width * height;
    let mut parents: Vec<usize> = (0..n).collect();

    fn find(parents: &mut [usize], mut x: usize) -> usize {
        while parents[x] != x {
            let p = parents[x];
            parents[x] = parents[p];
            x = parents[x];
        }
        x
    }

    fn unite(parents: &mut [usize], a: usize, b: usize) {
        let ra = find(parents, a);
        let rb = find(parents, b);
        if ra == rb {
            return;
        }
        if ra < rb {
            parents[rb] = ra;
        } else {
            parents[ra] = rb;
        }
    }

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let v = thresholded[idx];
            if v == 127 {
                continue;
            }

            if x > 0 {
                let nidx = idx - 1;
                if thresholded[nidx] == v {
                    unite(&mut parents, idx, nidx);
                }
            }
            if y > 0 {
                let nidx = idx - width;
                if thresholded[nidx] == v {
                    unite(&mut parents, idx, nidx);
                }
            }

            // Match the CCL shader's effective topology:
            // white is 8-connected, background is 4-connected.
            if v == 255 && y > 0 {
                if x > 0 {
                    let nidx = idx - width - 1;
                    if thresholded[nidx] == v {
                        unite(&mut parents, idx, nidx);
                    }
                }
                if x + 1 < width {
                    let nidx = idx - width + 1;
                    if thresholded[nidx] == v {
                        unite(&mut parents, idx, nidx);
                    }
                }
            }
        }
    }

    let mut labels = vec![0u32; n];
    for i in 0..n {
        if thresholded[i] == 127 {
            labels[i] = i as u32;
        } else {
            labels[i] = find(&mut parents, i) as u32;
        }
    }

    let mut union_sizes = vec![0u32; n];
    for i in 0..n {
        if thresholded[i] == 0 || thresholded[i] == 255 {
            let root = labels[i] as usize;
            if root < n {
                union_sizes[root] = union_sizes[root].saturating_add(1);
            }
        }
    }

    (labels, union_sizes)
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
        Self { device, settings }
    }

    pub fn detect(&self, image: DynamicImage) -> Result<Vec<DetectedTag>, ()> {
        if let Some(factor) = self.settings.decimate
            && !is_power_of_two(factor)
        {
            return Err(());
        }

        let decimate_factor = self.settings.decimate.unwrap_or(1) as u32;
        let aligned_input = crop_image_to_multiple(image.into_luma8(), 4 * decimate_factor)?;
        println!(
            "[detect] input={}x{} decimate_factor={} min_white_black_diff={} min_blob_size={}",
            aligned_input.width(),
            aligned_input.height(),
            decimate_factor,
            self.settings.min_white_black_diff,
            self.settings.min_blob_size
        );
        let input_gpu_image =
            crate::GPUImage::from_image_buffer(self.device.clone(), aligned_input);

        let decimated_image = match self.settings.decimate {
            Some(factor) => {
                let new_img = self.decimate(&input_gpu_image, factor);
                new_img
            }
            None => input_gpu_image.clone(),
        };
        println!(
            "[detect] decimated={}x{}",
            decimated_image.size.width, decimated_image.size.height
        );

        let (minmax_image, minmax_size) = self.minmax(&decimated_image);
        let filtered_minmax_image = self.filter_minmax(&minmax_image, minmax_size);
        let thresholded_image = self.threshold(
            &decimated_image,
            &filtered_minmax_image,
            minmax_size,
            self.settings.min_white_black_diff,
        );
        let thresholded = thresholded_image.data();
        let mut black = 0usize;
        let mut mid = 0usize;
        let mut white = 0usize;
        let mut black_large = 0usize;
        let mut white_large = 0usize;
        let mut max_black_union = 0u32;
        let mut max_white_union = 0u32;
        for &v in &thresholded {
            match v {
                0 => black += 1,
                127 => mid += 1,
                255 => white += 1,
                _ => {}
            }
        }
        println!(
            "[detect] thresholded pixels: black={} mid(127)={} white={} total={}",
            black,
            mid,
            white,
            thresholded.len()
        );
        let (cpu_ccl_labels, cpu_ccl_union_sizes) =
            ccl_from_thresholded_cpu(&thresholded, thresholded_image.size);
        let cpu_ccl_nonzero = cpu_ccl_union_sizes.iter().filter(|&&x| x > 0).count();
        let cpu_ccl_max = cpu_ccl_union_sizes.iter().copied().max().unwrap_or(0);
        println!(
            "[detect] ccl independent cpu: nonzero={} max={}",
            cpu_ccl_nonzero, cpu_ccl_max
        );
        let (cpu_ccl_labels_dual, cpu_ccl_union_sizes_dual) =
            ccl_from_thresholded_cpu_dual_connectivity(&thresholded, thresholded_image.size);
        let cpu_ccl_dual_nonzero = cpu_ccl_union_sizes_dual.iter().filter(|&&x| x > 0).count();
        let cpu_ccl_dual_max = cpu_ccl_union_sizes_dual.iter().copied().max().unwrap_or(0);
        println!(
            "[detect] ccl cpu dual-connectivity: nonzero={} max={}",
            cpu_ccl_dual_nonzero, cpu_ccl_dual_max
        );

        let labels = self.ccl_init(&thresholded_image);
        let labels_after_init =
            self.download_u32_buffer(&labels, thresholded_image.size.total_pixels());
        let width = thresholded_image.size.width as usize;
        let height = thresholded_image.size.height as usize;
        let mut fg_info_low_mismatch = 0usize;
        let mut left_bg_info_low_mismatch = 0usize;
        let mut right_bg_info_low_mismatch = 0usize;
        let mut fg_info_sample = Vec::new();
        for y in (0..height).step_by(2) {
            for x in (0..width).step_by(2) {
                if x + 1 >= width || y + 1 >= height {
                    continue;
                }
                let idx = y * width + x;
                let info = labels_after_init[idx + 1];
                let fg_low = (info & 0xFu32) as u8;
                let left_low = ((info >> 8) & 0xFu32) as u8;
                let right_low = ((info >> 16) & 0xFu32) as u8;

                let v0 = thresholded[idx];
                let v1 = thresholded[idx + 1];
                let v2 = thresholded[idx + width];
                let v3 = thresholded[idx + width + 1];

                let mut fg_expected = 0u8;
                if v0 == 255 {
                    fg_expected |= 1 << 0;
                }
                if v1 == 255 {
                    fg_expected |= 1 << 1;
                }
                if v2 == 255 {
                    fg_expected |= 1 << 2;
                }
                if v3 == 255 {
                    fg_expected |= 1 << 3;
                }

                let mut left_expected = 0u8;
                if v0 == 0 {
                    left_expected |= 1 << 0;
                }
                if v2 == 0 {
                    left_expected |= 1 << 2;
                }

                let mut right_expected = 0u8;
                if v1 == 0 {
                    right_expected |= 1 << 1;
                }
                if v3 == 0 {
                    right_expected |= 1 << 3;
                }

                if fg_low != fg_expected {
                    fg_info_low_mismatch += 1;
                    if fg_info_sample.len() < 8 {
                        fg_info_sample.push((idx as u32, fg_low, fg_expected, v0, v1, v2, v3));
                    }
                }
                if left_low != left_expected {
                    left_bg_info_low_mismatch += 1;
                }
                if right_low != right_expected {
                    right_bg_info_low_mismatch += 1;
                }
            }
        }
        let tile_count = (width / 2) * (height / 2);
        println!(
            "[detect] ccl init info-low compare: fg_mismatch={}/{} left_bg_mismatch={}/{} right_bg_mismatch={}/{}",
            fg_info_low_mismatch,
            tile_count,
            left_bg_info_low_mismatch,
            tile_count,
            right_bg_info_low_mismatch,
            tile_count
        );
        if !fg_info_sample.is_empty() {
            println!(
                "[detect] ccl init fg mismatch sample (idx,fg_low,fg_expected,v00,v10,v01,v11): {:?}",
                fg_info_sample
            );
        }
        self.ccl_compression(&labels, thresholded_image.size);
        self.ccl_merge(&labels, thresholded_image.size);
        self.ccl_compression(&labels, thresholded_image.size);

        let union_markers_size =
            self.new_zeroed_u32_storage_buffer(thresholded_image.size.total_pixels());
        self.ccl_final_labeling(&labels, &union_markers_size, thresholded_image.size);
        let union_sizes_gpu =
            self.download_u32_buffer(&union_markers_size, thresholded_image.size.total_pixels());
        let label_words = self.download_u32_buffer(&labels, thresholded_image.size.total_pixels());
        let label_words_canonical = canonicalize_labels_cpu(&label_words);
        let relabeled = label_words_canonical
            .iter()
            .zip(label_words.iter())
            .filter(|(a, b)| a != b)
            .count();
        println!(
            "[detect] ccl canonicalized labels on CPU: changed={}/{}",
            relabeled,
            label_words_canonical.len()
        );
        let nonzero_unions = union_sizes_gpu.iter().filter(|&&x| x > 0).count();
        let max_union = union_sizes_gpu.iter().copied().max().unwrap_or(0);
        println!(
            "[detect] ccl union sizes: nonzero={} max={}",
            nonzero_unions, max_union
        );

        let mut union_sizes_cpu = vec![0u32; thresholded_image.size.total_pixels()];
        for (i, &label) in label_words.iter().enumerate() {
            let label = label as usize;
            if label >= union_sizes_cpu.len() {
                continue;
            }
            if thresholded[i] == 0 || thresholded[i] == 255 {
                union_sizes_cpu[label] = union_sizes_cpu[label].saturating_add(1);
            }
        }
        let max_union_cpu = union_sizes_cpu.iter().copied().max().unwrap_or(0);
        let nonzero_unions_cpu = union_sizes_cpu.iter().filter(|&&x| x > 0).count();
        println!(
            "[detect] ccl cpu union sizes from labels: nonzero={} max={}",
            nonzero_unions_cpu, max_union_cpu
        );

        let max_white_union_gpu = label_words
            .iter()
            .enumerate()
            .filter_map(|(i, &l)| {
                if thresholded[i] != 255 {
                    return None;
                }
                let li = l as usize;
                (li < union_sizes_gpu.len()).then_some(union_sizes_gpu[li])
            })
            .max()
            .unwrap_or(0);
        let max_white_union_cpu = label_words
            .iter()
            .enumerate()
            .filter_map(|(i, &l)| {
                if thresholded[i] != 255 {
                    return None;
                }
                let li = l as usize;
                (li < union_sizes_cpu.len()).then_some(union_sizes_cpu[li])
            })
            .max()
            .unwrap_or(0);

        let cpu_white_max_independent = label_words
            .iter()
            .enumerate()
            .filter_map(|(i, _)| {
                if thresholded[i] != 255 {
                    return None;
                }
                let li = cpu_ccl_labels[i] as usize;
                (li < cpu_ccl_union_sizes.len()).then_some(cpu_ccl_union_sizes[li])
            })
            .max()
            .unwrap_or(0);
        let cpu_white_max_dual = label_words
            .iter()
            .enumerate()
            .filter_map(|(i, _)| {
                if thresholded[i] != 255 {
                    return None;
                }
                let li = cpu_ccl_labels_dual[i] as usize;
                (li < cpu_ccl_union_sizes_dual.len()).then_some(cpu_ccl_union_sizes_dual[li])
            })
            .max()
            .unwrap_or(0);
        println!(
            "[detect] ccl white-max compare: gpu_union[label]={} gpu_recount[label]={} cpu_independent={} cpu_dual={}",
            max_white_union_gpu, max_white_union_cpu, cpu_white_max_independent, cpu_white_max_dual
        );
        let mut label_word_mismatch = 0usize;
        let mut label_bit_mismatch = 0u64;
        for i in 0..label_words.len() {
            let gpu = label_words[i];
            let cpu = cpu_ccl_labels_dual[i];
            if gpu != cpu {
                label_word_mismatch += 1;
                label_bit_mismatch += (gpu ^ cpu).count_ones() as u64;
            }
        }
        let mut union_word_mismatch = 0usize;
        let mut union_bit_mismatch = 0u64;
        for i in 0..union_sizes_gpu.len() {
            let gpu = union_sizes_gpu[i];
            let cpu = cpu_ccl_union_sizes_dual[i];
            if gpu != cpu {
                union_word_mismatch += 1;
                union_bit_mismatch += (gpu ^ cpu).count_ones() as u64;
            }
        }
        let mut pixel_size_mismatch = 0usize;
        let mut pixel_size_mismatch_black = 0usize;
        let mut pixel_size_mismatch_white = 0usize;
        let mut pixel_mismatch_sample = Vec::new();
        for i in 0..thresholded.len() {
            if thresholded[i] == 127 {
                continue;
            }
            let gl = label_words[i] as usize;
            let cl = cpu_ccl_labels_dual[i] as usize;
            let gpu_size = if gl < union_sizes_gpu.len() {
                union_sizes_gpu[gl]
            } else {
                0
            };
            let cpu_size = if cl < cpu_ccl_union_sizes_dual.len() {
                cpu_ccl_union_sizes_dual[cl]
            } else {
                0
            };
            if gpu_size != cpu_size {
                pixel_size_mismatch += 1;
                if thresholded[i] == 0 {
                    pixel_size_mismatch_black += 1;
                } else if thresholded[i] == 255 {
                    pixel_size_mismatch_white += 1;
                }
                if pixel_mismatch_sample.len() < 8 {
                    pixel_mismatch_sample.push((
                        i,
                        thresholded[i],
                        gl as u32,
                        gpu_size,
                        cl as u32,
                        cpu_size,
                    ));
                }
            }
        }
        println!(
            "[detect] ccl compare labels(bitwise): word_mismatch={}/{} bit_mismatch={}",
            label_word_mismatch,
            label_words.len(),
            label_bit_mismatch
        );
        println!(
            "[detect] ccl compare union_sizes(bitwise): word_mismatch={}/{} bit_mismatch={}",
            union_word_mismatch,
            union_sizes_gpu.len(),
            union_bit_mismatch
        );
        println!(
            "[detect] ccl compare per-pixel component-size: mismatch={} black={} white={}",
            pixel_size_mismatch, pixel_size_mismatch_black, pixel_size_mismatch_white
        );
        if !pixel_mismatch_sample.is_empty() {
            println!(
                "[detect] ccl per-pixel mismatch sample (idx,v,gpu_label,gpu_size,cpu_label,cpu_size): {:?}",
                pixel_mismatch_sample
            );
        }

        let gpu_ccl_failed = fg_info_low_mismatch > 0
            || (max_white_union_gpu == 0 && cpu_white_max_dual > 0)
            || (white > 0 && pixel_size_mismatch_white * 100 > white * 5);
        if gpu_ccl_failed {
            eprintln!(
                "[detect] ERROR: GPU CCL validation failed (fg_mismatch={} gpu_white_max={} cpu_white_max={} white_pixel_size_mismatch={}/{})",
                fg_info_low_mismatch,
                max_white_union_gpu,
                cpu_white_max_dual,
                pixel_size_mismatch_white,
                white
            );
            return Err(());
        }

        let labels_for_blob_diff = labels.clone();
        let union_sizes = union_sizes_gpu;

        let labels_out_of_range = label_words
            .iter()
            .filter(|&&l| l as usize >= union_sizes.len())
            .count();
        let large_label_pixels = label_words
            .iter()
            .filter(|&&l| {
                (l as usize) < union_sizes.len()
                    && union_sizes[l as usize] >= self.settings.min_blob_size
            })
            .count();
        println!(
            "[detect] labels: out_of_range={} pixels_with_union(label)>={} => {}",
            labels_out_of_range, self.settings.min_blob_size, large_label_pixels
        );
        if !label_words.is_empty() {
            let mut sample = Vec::new();
            for i in 0..usize::min(8, label_words.len()) {
                let l = label_words[i] as usize;
                let us = if l < union_sizes.len() {
                    union_sizes[l]
                } else {
                    u32::MAX
                };
                sample.push((i, label_words[i], us));
            }
            println!(
                "[detect] label/union sample (idx,label,union[label]): {:?}",
                sample
            );
        }
        for i in 0..thresholded.len() {
            let l = label_words[i] as usize;
            let union = if l < union_sizes.len() {
                union_sizes[l]
            } else {
                0
            };
            let large = union >= self.settings.min_blob_size;
            if !large {
                match thresholded[i] {
                    0 => max_black_union = max_black_union.max(union),
                    255 => max_white_union = max_white_union.max(union),
                    _ => {}
                }
                continue;
            }
            match thresholded[i] {
                0 => {
                    black_large += 1;
                    max_black_union = max_black_union.max(union);
                }
                255 => {
                    white_large += 1;
                    max_white_union = max_white_union.max(union);
                }
                _ => {}
            }
        }
        println!(
            "[detect] large-component polarity pixels: black={} white={}",
            black_large, white_large
        );
        println!(
            "[detect] max union by polarity: black={} white={}",
            max_black_union, max_white_union
        );

        let blob_diff_words_per_point = 6usize;
        let blob_diff_points_per_offset = (thresholded_image.size.width as usize - 2)
            * (thresholded_image.size.height as usize - 2);
        let blob_diff_total_points = (blob_diff_points_per_offset * 4) as u32;
        let blob_diff_out = self
            .new_u32_storage_buffer(blob_diff_total_points as usize * blob_diff_words_per_point);
        self.blob_diff(
            &thresholded_image,
            &labels_for_blob_diff,
            &union_markers_size,
            &blob_diff_out,
            self.settings.min_blob_size,
        );

        let mut eligible_centers = 0usize;
        let mut raw_bw_adjacencies = 0usize;
        let mut bw_with_large_rep0 = 0usize;
        let mut bw_with_large_rep1 = 0usize;
        let mut potential_connections = 0usize;
        for y in 1..(height.saturating_sub(1)) {
            for x in 1..(width.saturating_sub(1)) {
                let idx = y * width + x;
                let v0 = thresholded[idx];
                let rep0 = label_words[idx] as usize;
                let rep0_large =
                    rep0 < union_sizes.len() && union_sizes[rep0] >= self.settings.min_blob_size;
                if v0 == 127 || rep0 >= union_sizes.len() || !rep0_large {
                    continue;
                }
                eligible_centers += 1;
                let neighbors = [(1isize, 0isize), (1, 1), (0, 1), (-1, 1)];
                for (dx, dy) in neighbors {
                    let nx = (x as isize + dx) as usize;
                    let ny = (y as isize + dy) as usize;
                    let nidx = ny * width + nx;
                    let v1 = thresholded[nidx];
                    let rep1 = label_words[nidx] as usize;
                    if v0 as u16 + v1 as u16 == 255 {
                        raw_bw_adjacencies += 1;
                        if rep0_large {
                            bw_with_large_rep0 += 1;
                        }
                        if rep1 < union_sizes.len()
                            && union_sizes[rep1] >= self.settings.min_blob_size
                        {
                            bw_with_large_rep1 += 1;
                            if rep0_large {
                                potential_connections += 1;
                            }
                        }
                    }
                }
            }
        }
        println!(
            "[detect] blob_diff CPU precheck: eligible_centers={} raw_bw_adjacencies={} bw_with_large_rep0={} bw_with_large_rep1={} potential_connections={}",
            eligible_centers,
            raw_bw_adjacencies,
            bw_with_large_rep0,
            bw_with_large_rep1,
            potential_connections
        );

        let blob_diff_count = self.new_zeroed_u32_counter_buffer();
        self.count_nonzero_blob_diff_points(
            &blob_diff_out,
            &blob_diff_count,
            blob_diff_total_points,
        );
        let blob_diff_compacted_size = self.read_counter(&blob_diff_count);
        println!(
            "[detect] blob_diff nonzero points: {}/{}",
            blob_diff_compacted_size, blob_diff_total_points
        );

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
        println!(
            "[detect] blob_diff compacted points: {}",
            blob_diff_filtered_size
        );
        if blob_diff_filtered_size > 0 {
            let sample = self.download_u32_buffer(
                &blob_diff_compacted,
                usize::min(3, blob_diff_filtered_size as usize) * blob_diff_words_per_point,
            );
            println!("[detect] blob_diff first points (u32 words): {:?}", sample);
        }

        let blob_diff_sort_points = next_power_of_two(blob_diff_filtered_size);
        println!(
            "[detect] blob_diff sort points (pow2): {}",
            blob_diff_sort_points
        );
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
        println!("[detect] blob extents: {}", blob_extent_count_value);
        if blob_extent_count_value > 0 {
            let sample = self.download_u32_buffer(
                &blob_extent,
                usize::min(2, blob_extent_count_value as usize) * blob_extent_words_per_extent,
            );
            println!("[detect] first blob extent words: {:?}", sample);
        }

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
        println!(
            "[detect] selected extents: {}/{} selected points: {}",
            selected_blob_extent_count_value,
            blob_extent_count_value,
            selected_blob_point_count_value
        );

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
        if selected_blob_point_count_value > 0 {
            let sample = self.download_u32_buffer(
                &selected_blob_points,
                usize::min(3, selected_blob_point_count_value as usize)
                    * selected_blob_point_words_per_point,
            );
            println!(
                "[detect] selected blob points (unsorted) sample: {:?}",
                sample
            );
        }

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
        if selected_blob_point_count_value > 0 {
            let sample = self.download_u32_buffer(
                &selected_blob_sorted_points,
                usize::min(3, selected_blob_point_count_value as usize)
                    * selected_blob_point_words_per_point,
            );
            println!(
                "[detect] selected blob points (sorted) sample: {:?}",
                sample
            );
        }

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
        println!(
            "[detect] valid peaks: {}/{}",
            peak_count_value, selected_blob_point_count_value
        );

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
        println!("[detect] compacted peaks: {}", compacted_peak_count_value);
        if compacted_peak_count_value > 0 {
            let sample = self.download_u32_buffer(
                &compacted_peaks,
                usize::min(3, compacted_peak_count_value as usize) * peak_words_per_peak,
            );
            println!("[detect] compacted peak sample: {:?}", sample);
        }

        let peak_sort_points = next_power_of_two(compacted_peak_count_value);
        println!("[detect] peak sort points (pow2): {}", peak_sort_points);
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
        println!("[detect] peak extents: {}", peak_extent_count_value);
        if peak_extent_count_value > 0 {
            let peak_extent_words = self.download_u32_buffer(
                &peak_extents,
                peak_extent_count_value as usize * peak_extent_words_per_extent,
            );
            let filtered_extent_words = self.download_u32_buffer(
                &filtered_blob_extent,
                usize::max(1, blob_extent_count_value as usize) * blob_extent_words_per_extent,
            );
            let mut gate_peak_count = 0u32;
            let mut gate_blob_count = 0u32;
            let mut gate_blob_index = 0u32;
            let mut gate_total = 0u32;
            for i in 0..(peak_extent_count_value as usize) {
                gate_total += 1;
                let base = i * peak_extent_words_per_extent;
                let blob_index = peak_extent_words[base] as usize;
                let peak_count = peak_extent_words[base + 2];
                if blob_index >= blob_extent_count_value as usize {
                    gate_blob_index += 1;
                    continue;
                }
                if peak_count < 4 {
                    gate_peak_count += 1;
                    continue;
                }
                let blob_base = blob_index * blob_extent_words_per_extent;
                let blob_count = filtered_extent_words[blob_base + 7];
                if blob_count < 8 {
                    gate_blob_count += 1;
                    continue;
                }
            }
            println!(
                "[detect] fit_quads pre-gates: total={} fail_blob_index={} fail_peak_count={} fail_blob_count<8={}",
                gate_total, gate_blob_index, gate_peak_count, gate_blob_count
            );
        }

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
        println!("[detect] fitted quads: {}", fitted_quad_count_value);
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
            let first_bits = &bits_words[0..usize::min(bits_words.len(), 64)];
            let ones = first_bits.iter().filter(|&&b| b != 0).count();
            println!(
                "[detect] decode bits sample: {} words, ones_in_first_tag={}",
                bits_words.len(),
                ones
            );
            detected_tags =
                self.build_detected_tags(&fitted_quad_words, fitted_quad_count_value, &bits_words);
        }
        println!("[detect] detected tags: {}", detected_tags.len());
        if let Some(first) = detected_tags.first() {
            let ones = first.bits_with_border.iter().filter(|&&b| b != 0).count();
            println!(
                "[detect] first tag: quad={} blob={} score={:.4} ones={}/{}",
                first.quad_index,
                first.blob_index,
                first.score,
                ones,
                first.bits_with_border.len()
            );
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
        let mut best_distance = u32::MAX;
        let mut best_id = 0u32;
        let mut best_rotation = 0u8;

        for (id, code) in APRILTAG_36H11_CODES.iter().copied().enumerate() {
            let mut rotated_code = code;
            for rotation in 0..4u8 {
                let distance = (payload_code ^ rotated_code).count_ones();
                if distance < best_distance {
                    best_distance = distance;
                    best_id = id as u32;
                    best_rotation = rotation;
                }
                rotated_code = Self::rotate_code_ccw(rotated_code, Self::APRILTAG_MARKER_SIZE);
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

    fn create_compute_pipeline(&self, module_bytes: &[u8]) -> Arc<ComputePipeline> {
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
        .expect("failed to create pipeline layout");

        ComputePipeline::new(
            self.device.device.clone(),
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
        let module_bytes = include_bytes!(compute_shader_path!("01-threshold-decimate"));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);

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
        let module_bytes = include_bytes!(compute_shader_path!("02-threshold-minmax"));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);

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
        let module_bytes = include_bytes!(compute_shader_path!("03-threshold-filter-minmax"));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);

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
        let module_bytes = include_bytes!(compute_shader_path!("04-threshold-threshold"));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);

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
        let module_bytes = include_bytes!(compute_shader_path!("05-ccl-init"));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);

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
        let module_bytes = include_bytes!(compute_shader_path!("06-ccl-compression"));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);

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
        let module_bytes = include_bytes!(compute_shader_path!("07-ccl-merge"));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);

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
        let module_bytes = include_bytes!(compute_shader_path!("08-ccl-final-labeling"));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);

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
        let module_bytes = include_bytes!(compute_shader_path!("09-blob-blob-diff"));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);
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
        let module_bytes = include_bytes!(compute_shader_path!(
            "10-select-count-nonzero-blob-diff-points"
        ));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);
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
        let module_bytes = include_bytes!(compute_shader_path!(
            "11-select-filter-nonzero-blob-diff-points"
        ));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);
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
        let module_bytes = include_bytes!(compute_shader_path!("12-sort-prepare-blob-diff-points"));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);
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

        let module_bytes = include_bytes!(compute_shader_path!(
            "13-sort-bitonic-sort-blob-diff-points"
        ));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);
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
        let module_bytes =
            include_bytes!(compute_shader_path!("14-filter-build-blob-pair-extents"));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);
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
        let module_bytes =
            include_bytes!(compute_shader_path!("15-filter-filter-blob-pair-extents"));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);
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
        let module_bytes = include_bytes!(compute_shader_path!(
            "16-filter-rewrite-selected-blob-points-with-theta"
        ));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);
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
        let module_bytes =
            include_bytes!(compute_shader_path!("17-sort-prepare-selected-blob-points"));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);
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

        let module_bytes = include_bytes!(compute_shader_path!(
            "18-sort-bitonic-sort-selected-blob-points"
        ));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);
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
        let module_bytes = include_bytes!(compute_shader_path!("19-filter-build-line-fit-points"));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);
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
        let module_bytes =
            include_bytes!(compute_shader_path!("20-filter-fit-line-errors-and-peaks"));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);
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
        let module_bytes = include_bytes!(compute_shader_path!("21-select-count-valid-peaks"));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);
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
        let module_bytes = include_bytes!(compute_shader_path!("22-select-filter-valid-peaks"));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);
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
        let module_bytes = include_bytes!(compute_shader_path!("23-sort-prepare-peaks"));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);
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

        let module_bytes = include_bytes!(compute_shader_path!("24-sort-bitonic-sort-peaks"));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);
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
        let module_bytes = include_bytes!(compute_shader_path!("25-filter-build-peak-extents"));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);
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
        let module_bytes = include_bytes!(compute_shader_path!("26-filter-fit-quads"));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);
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
        let module_bytes = include_bytes!(compute_shader_path!("27-decode-prepare-decode-quads"));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);
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
        let module_bytes = include_bytes!(compute_shader_path!("28-decode-extract-candidate-bits"));
        let compute_pipeline = self.create_compute_pipeline(module_bytes);
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
