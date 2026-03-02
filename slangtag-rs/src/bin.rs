use slangtag::{
    ComputeDevice,
    detect::{DetectionSettings, Detector},
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let image_path = args
        .get(1)
        .expect("Please provide an image path as the first argument");
    let min_blob_size = args
        .get(2)
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(25);
    let image = image::open(image_path).expect("Failed to open image");

    let dev = ComputeDevice::new_default();
    let det = Detector::new(
        dev,
        DetectionSettings {
            decimate: Some(2),
            min_blob_size,
            ..Default::default()
        },
    );

    let tags = det.detect(image).expect("Detection failed");
    println!("Detected {} tag candidates", tags.len());
    for tag in &tags {
        println!(
            "quad={} blob={} score={:.4} reversed={} id={:?}",
            tag.quad_index, tag.blob_index, tag.score, tag.reversed_border, tag.id
        );
    }
}
