use slangtag::ComputeDevice;
use slangtag_cam::{CameraConfig, CameraTagStream, DetectionSettings};
use std::time::{Duration, Instant};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: {} <device_path> <width> <height> [fps] [max_frames]",
            args.first().map_or("print_tags", String::as_str)
        );
        std::process::exit(2);
    }

    let device_path = args[1].clone();
    let width = args[2]
        .parse::<u32>()
        .expect("width must be a positive integer");
    let height = args[3]
        .parse::<u32>()
        .expect("height must be a positive integer");
    let fps = args.get(4).and_then(|s| s.parse::<u32>().ok());
    let max_frames = args.get(5).and_then(|s| s.parse::<u64>().ok());

    let device = ComputeDevice::new_default();
    let mut config = CameraConfig::new(device_path, width, height);
    config.fps = fps;
    config.timing_debug = true;
    config.timing_every_n_frames = 30;
    config.use_device_local_input = true;
    let settings = DetectionSettings {
        decimate: Some(2),
        min_white_black_diff: 70,
        ..Default::default()
    };

    let mut stream =
        CameraTagStream::new(device, config, settings).expect("failed to start camera tag stream");

    let mut frame_index = 0u64;
    let start_time = Instant::now();
    let mut fps_window_start = start_time;
    let mut fps_window_frames = 0u64;
    for item in &mut stream {
        frame_index = frame_index.wrapping_add(1);
        fps_window_frames = fps_window_frames.wrapping_add(1);

        match item {
            Ok(tags) => {
                if !tags.is_empty() {
                    println!("frame {frame_index}: {} tag(s)", tags.len());
                    for tag in &tags {
                        println!(
                            "  id={:?} quad_index={} blob_index={} reversed={} score={:.3} corners={:?}",
                            tag.id,
                            tag.quad_index,
                            tag.blob_index,
                            tag.reversed_border,
                            tag.score,
                            tag.corners
                        );
                    }
                }
            }
            Err(err) => {
                eprintln!("frame {frame_index}: error: {err}");
                break;
            }
        }

        let now = Instant::now();
        let window_elapsed = now.duration_since(fps_window_start);
        if window_elapsed >= Duration::from_secs(1) {
            let fps = fps_window_frames as f64 / window_elapsed.as_secs_f64();
            println!("fps: {:.2}", fps);
            fps_window_start = now;
            fps_window_frames = 0;
        }

        if let Some(limit) = max_frames
            && frame_index >= limit
        {
            break;
        }
    }

    let total_elapsed = start_time.elapsed();
    if frame_index > 0 && total_elapsed > Duration::ZERO {
        let avg_fps = frame_index as f64 / total_elapsed.as_secs_f64();
        println!(
            "processed {frame_index} frame(s) in {:.2}s (avg fps: {:.2})",
            total_elapsed.as_secs_f64(),
            avg_fps
        );
    }
}
