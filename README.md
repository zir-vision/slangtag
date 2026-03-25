
## Licensing

This project utilized LLMs heavily but used multiple existing projects as reference. Copyright law doesn't seem to be clear on how to handle this, but I list the licenses of those projects here for transparency.

- Most of the GPU AprilTag pipeline code up to tag decoding is based off [971's CUDA AprilTag implementation](https://github.com/frc971/971-Robot-Code), which is licensed under the Apache License 2.0.
- Tag decoding is based off of [OpenCV](https://github.com/opencv/opencv/blob/4.x/modules/objdetect/src/aruco/aruco_detector.cpp), which is licensed under the Apache License 2.0.
- The radix sort Slang implementation is copied from [jaesung-cs/vulkan_radix_sort](https://github.com/jaesung-cs/vulkan_radix_sort) under the MIT license and modified a little bit to work with Intel GPUs.

That being said, this project as a whole is licensed under the Lesser GNU Public License v3.0, which is a copyleft license that requires derivative works to also be licensed under the same terms. This means that if you use or modify this code, you must also release your changes under the same license.