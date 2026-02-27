use image::{DynamicImage, GenericImageView};

use crate::ComputeDevice;


pub struct Detector {
    device: ComputeDevice
}

impl Detector {
    pub fn new(device: ComputeDevice) -> Self {
        Self { device }
    }

    pub fn detect(&self, image: DynamicImage) -> Result<(), ()> {
        
        Ok(())
    }
}