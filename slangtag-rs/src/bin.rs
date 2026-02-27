use slangtag::{ComputeDevice, detect::Detector};


fn main() {
    
    let dev = ComputeDevice::new_default();
    let det = Detector::new(dev);
    
}
