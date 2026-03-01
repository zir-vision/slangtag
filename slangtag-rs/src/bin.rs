use slangtag::{ComputeDevice, detect::{DetectionSettings, Detector}};


fn main() {
    let args: Vec<String> = std::env::args().collect();
    let image_path = args.get(1).expect("Please provide an image path as the first argument");
    let image = image::open(image_path).expect("Failed to open image");


    let dev = ComputeDevice::new_default();
    let det = Detector::new(dev, DetectionSettings {
        decimate: Some(2),
        ..Default::default()
    });
    
    det.detect(image).expect("Detection failed");

}
