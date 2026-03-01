use image::{DynamicImage, GrayImage, ImageBuffer};

use crate::ComputeDevice;


pub struct Detector {
    device: ComputeDevice
}

impl Detector {
    pub fn new(device: ComputeDevice) -> Self {
        Self { device }
    }

    pub fn detect(&self, image: DynamicImage) -> Result<(), ()> {
        let gray = image.into_luma8();
        Ok(())
    }
}