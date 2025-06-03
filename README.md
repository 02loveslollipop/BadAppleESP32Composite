# BadAppleESP32Composite
> A new implementation of the Bad Apple video for ESP32, using a composite video output, but with a way better performance than the original implementation and audio support.

---

## Overview
This project is a complete reimplementation of the [BadAppleESP32CodeMaker](https://github.com/02loveslollipop/BadAppleESP32CodeMaker) but with a focus on performance on encoding the video and audio support.

To achieve this, the video encoding was refactored from a monolithic (and mostly spaghetti) and sequential implementation to a modular and parallel implementation that uses Numba instead of Cython to speed up the video encoding process and support for CUDA GPU acceleration using a separate CuPy implementation.

## Features
- Full CPython implementation with Numba for CPU acceleration that doesn't require Cython compilation
- Support for CUDA GPU acceleration using CuPy
- Improved perfomance by parallelizing the video encoding process by making independent the encoding of each line of the video while keeping the same delta encoding and RLE compression logic.
- Designed to be potentially compatible with Motion Compensation encoding in the future
- Full retrocompatibility with the original BadAppleESP32CodeMaker ESP32 implementation if no audio is used
