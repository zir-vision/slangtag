use slangtag::{
    ComputeDevice,
    detect::{DetectionSettings, Detector},
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

    let dev = ComputeDevice::new_default();
    let det = Detector::new(
        dev,
        DetectionSettings {
            decimate: Some(2),
            ..Default::default()
        },
    );

    let mut total_wall_ms = 0.0f64;
    let mut total_gpu_ms = 0.0f64;
    let mut tag_count = 0usize;
    let mut ordered_span_names = Vec::<String>::new();
    let mut span_sums_ms = Vec::<f64>::new();
    let mut span_counts = Vec::<usize>::new();

    for _ in 0..runs {
        let start = Instant::now();
        let (tags, timing_report) = det
            .detect_gray_with_timing(gray_image.clone())
            .expect("Detection failed");
        total_wall_ms += start.elapsed().as_secs_f64() * 1_000.0;
        total_gpu_ms += timing_report.total_ms;
        tag_count = tags.len();

        for span in timing_report.spans {
            let index = if let Some(existing_index) = ordered_span_names
                .iter()
                .position(|name| name == &span.name)
            {
                existing_index
            } else {
                ordered_span_names.push(span.name);
                span_sums_ms.push(0.0);
                span_counts.push(0);
                ordered_span_names.len() - 1
            };
            span_sums_ms[index] += span.elapsed_ms;
            span_counts[index] += 1;
        }
    }

    let avg_wall_ms = total_wall_ms / runs as f64;
    let avg_gpu_ms = total_gpu_ms / runs as f64;
    let averaged_spans = ordered_span_names
        .into_iter()
        .enumerate()
        .map(|(index, name)| {
            let count = span_counts.get(index).copied().unwrap_or(1);
            let sum_ms = span_sums_ms.get(index).copied().unwrap_or(0.0);
            (name, sum_ms / count as f64)
        })
        .collect::<Vec<_>>();

    println!(
        "\x1b[1;36mAveraged over {} run(s) for {}\x1b[0m",
        runs, image_path
    );
    println!("Detected {} tags", tag_count);
    println!(
        "Average wall time: {:>9.3} ms | Average GPU timed shaders: {:>9.3} ms",
        avg_wall_ms, avg_gpu_ms
    );
    println!("GPU shader timings (average):");
    for (name, avg_ms) in averaged_spans {
        println!("  {:<52} {}", name, colorize_ms(avg_ms, avg_gpu_ms));
    }
    println!(
        "  {:<52} {:>9.3} ms",
        "total timed shader execution", avg_gpu_ms
    );
}
