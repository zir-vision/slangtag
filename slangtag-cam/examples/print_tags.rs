use slangtag::ComputeDevice;
use slangtag_cam::{CameraConfig, CameraTagStream, DetectionSettings};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: {} <device_path> <width> <height> [max_frames]",
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
    let max_frames = args.get(4).and_then(|s| s.parse::<u64>().ok());

    let device = ComputeDevice::new_default();
    let config = CameraConfig::new(device_path, width, height);
    let settings = DetectionSettings::default();

    let mut stream = CameraTagStream::new(device, config, settings)
        .expect("failed to start camera tag stream");

    let mut frame_index = 0u64;
    for item in &mut stream {
        frame_index = frame_index.wrapping_add(1);

        match item {
            Ok(tags) => {
                if tags.is_empty() {
                    println!("frame {frame_index}: no tags");
                } else {
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

        if let Some(limit) = max_frames
            && frame_index >= limit
        {
            break;
        }
    }
}
