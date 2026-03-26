use ash::vk;
use slangtag::{
    ComputeCommandContext, ComputeDevice, Size,
    detect::{DetectionSettings, DetectionTimingMode, Detector},
};
use std::time::Instant;

fn colorize_ms(ms: f64, total_ms: f64) -> String {
    let ratio = if total_ms > 0.0 { ms / total_ms } else { 0.0 };
    let color = if ratio >= 0.25 {
        196
    } else if ratio >= 0.10 {
        220
    } else {
        46
    };
    format!("\x1b[38;5;{color}m{:>9.3} ms\x1b[0m", ms)
}

fn crop_image_to_multiple(image: image::GrayImage, multiple: u32) -> Result<image::GrayImage, ()> {
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let image_path = args
        .get(1)
        .expect("Please provide an image path as the first argument");
    let runs = args
        .get(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(5);
    assert!(runs > 0, "Run count must be greater than zero");

    let image = image::open(image_path).expect("Failed to open image");
    let gray_image = image.to_luma8();

    let settings = DetectionSettings {
        timing_mode: DetectionTimingMode::Detailed,
        decimate: Some(2),
        ..Default::default()
    };
    let decimate_factor = settings.decimate.unwrap_or(1) as u32;
    let aligned_input = crop_image_to_multiple(gray_image, 4 * decimate_factor)
        .expect("image is too small after decimate alignment");
    let input_size = Size::new(aligned_input.width(), aligned_input.height());

    let dev = ComputeDevice::new_default();
    let mut command_context: ComputeCommandContext = dev.create_command_context();
    let input_gpu_buffer = dev.upload_buffer(
        &mut command_context,
        aligned_input.as_raw(),
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        true,
    );

    let det =
        Detector::new(dev, settings, input_size).expect("failed to initialize fixed-size detector");

    let mut total_wall_ms = 0.0f64;
    let mut total_gpu_ms = 0.0f64;
    let mut total_cpu_ms = 0.0f64;
    let mut total_timed_ms = 0.0f64;
    let mut tag_count = 0usize;
    let mut ordered_gpu_span_names = Vec::<String>::new();
    let mut gpu_span_sums_ms = Vec::<f64>::new();
    let mut gpu_span_counts = Vec::<usize>::new();
    let mut ordered_cpu_span_names = Vec::<String>::new();
    let mut cpu_span_sums_ms = Vec::<f64>::new();
    let mut cpu_span_counts = Vec::<usize>::new();

    for _ in 0..runs {
        let start = Instant::now();
        let output = det
            .detect_buffer(
                &mut command_context,
                input_gpu_buffer.descriptor(),
                input_size,
            )
            .expect("Detection failed");
        let timing_report = output.timing;
        total_wall_ms += start.elapsed().as_secs_f64() * 1_000.0;
        total_gpu_ms += timing_report.total_ms;
        total_cpu_ms += timing_report.cpu_total_ms;
        total_timed_ms += timing_report.end_to_end_ms;
        tag_count = output.tags.len();

        for span in timing_report.spans {
            let index = if let Some(existing_index) = ordered_gpu_span_names
                .iter()
                .position(|name| name == &span.name)
            {
                existing_index
            } else {
                ordered_gpu_span_names.push(span.name);
                gpu_span_sums_ms.push(0.0);
                gpu_span_counts.push(0);
                ordered_gpu_span_names.len() - 1
            };
            gpu_span_sums_ms[index] += span.elapsed_ms;
            gpu_span_counts[index] += 1;
        }

        for span in timing_report.cpu_spans {
            let index = if let Some(existing_index) = ordered_cpu_span_names
                .iter()
                .position(|name| name == &span.name)
            {
                existing_index
            } else {
                ordered_cpu_span_names.push(span.name);
                cpu_span_sums_ms.push(0.0);
                cpu_span_counts.push(0);
                ordered_cpu_span_names.len() - 1
            };
            cpu_span_sums_ms[index] += span.elapsed_ms;
            cpu_span_counts[index] += 1;
        }
    }

    let avg_wall_ms = total_wall_ms / runs as f64;
    let avg_gpu_ms = total_gpu_ms / runs as f64;
    let avg_cpu_ms = total_cpu_ms / runs as f64;
    let avg_timed_ms = total_timed_ms / runs as f64;
    let averaged_gpu_spans = ordered_gpu_span_names
        .into_iter()
        .enumerate()
        .map(|(index, name)| {
            let count = gpu_span_counts.get(index).copied().unwrap_or(1);
            let sum_ms = gpu_span_sums_ms.get(index).copied().unwrap_or(0.0);
            (name, sum_ms / count as f64)
        })
        .collect::<Vec<_>>();
    let averaged_cpu_spans = ordered_cpu_span_names
        .into_iter()
        .enumerate()
        .map(|(index, name)| {
            let count = cpu_span_counts.get(index).copied().unwrap_or(1);
            let sum_ms = cpu_span_sums_ms.get(index).copied().unwrap_or(0.0);
            (name, sum_ms / count as f64)
        })
        .collect::<Vec<_>>();

    println!(
        "\x1b[1;36mAveraged over {} run(s) for {}\x1b[0m",
        runs, image_path
    );
    println!("Detected {} tags", tag_count);
    println!(
        "Average wall time: {:>9.3} ms | Average GPU timed: {:>9.3} ms | Average CPU timed: {:>9.3} ms | Average timed total: {:>9.3} ms",
        avg_wall_ms, avg_gpu_ms, avg_cpu_ms, avg_timed_ms
    );
    println!("GPU shader timings (average):");
    for (name, avg_ms) in averaged_gpu_spans {
        println!("  {:<52} {}", name, colorize_ms(avg_ms, avg_gpu_ms));
    }
    println!(
        "  {:<52} {:>9.3} ms",
        "total timed shader execution", avg_gpu_ms
    );
    println!("CPU timings (average):");
    for (name, avg_ms) in averaged_cpu_spans {
        println!("  {:<52} {}", name, colorize_ms(avg_ms, avg_cpu_ms));
    }
    println!(
        "  {:<52} {:>9.3} ms",
        "total timed cpu execution", avg_cpu_ms
    );
    println!(
        "  {:<52} {:>9.3} ms",
        "total timed cpu+gpu execution", avg_timed_ms
    );
}
