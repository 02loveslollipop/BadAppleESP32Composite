"""
GPU-based 1-bit video conversion with dithering support.

This module provides GPU-optimized implementations for converting RGB/Grayscale
video to 1-bit with various dithering algorithms using CuPy for CUDA acceleration.
"""

import numpy as np
from typing import Optional, Literal, Dict

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None


def convert_1bit_gpu(frames: np.ndarray, 
                     threshold: int = 128,
                     dither_method: Optional[Literal['none', 'floyd-steinberg', 'ordered']] = 'none',
                     dither_pattern: Literal['bayer2x2', 'bayer4x4', 'bayer8x8'] = 'bayer4x4',
                     device_id: int = 0) -> np.ndarray:
    """
    Convert video frames to 1-bit using GPU processing.
    
    Args:
        frames: Input video frames (N, H, W) or (N, H, W, C)
        threshold: Threshold value for binary conversion (0-255)
        dither_method: Dithering method ('none', 'floyd-steinberg', 'ordered')
        dither_pattern: Pattern for ordered dithering
        device_id: CUDA device ID to use
        
    Returns:
        1-bit video frames as uint8 array (0 or 1 values)
        
    Raises:
        RuntimeError: If CUDA is not available
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available. Install CuPy for GPU acceleration.")
    
    # Set device
    with cp.cuda.Device(device_id):
        # Transfer to GPU
        frames_gpu = cp.asarray(frames)
        
        # Convert to grayscale if needed
        if len(frames_gpu.shape) == 4 and frames_gpu.shape[-1] == 3:
            grayscale_frames = rgb_to_grayscale_gpu(frames_gpu)
        elif len(frames_gpu.shape) == 4 and frames_gpu.shape[-1] == 1:
            grayscale_frames = frames_gpu.squeeze(-1)
        else:
            grayscale_frames = frames_gpu
        
        # Apply conversion method
        if dither_method == 'none':
            result_gpu = simple_threshold_gpu(grayscale_frames, threshold)
        elif dither_method == 'floyd-steinberg':
            result_gpu = floyd_steinberg_dither_gpu(grayscale_frames, threshold)
        elif dither_method == 'ordered':
            result_gpu = ordered_dither_gpu(grayscale_frames, threshold, dither_pattern)
        else:
            raise ValueError(f"Unknown dither method: {dither_method}")
        
        # Transfer back to CPU
        return cp.asnumpy(result_gpu)


def rgb_to_grayscale_gpu(frames_gpu: 'cp.ndarray') -> 'cp.ndarray':
    """Convert RGB frames to grayscale using GPU."""
    weights = cp.array([0.299, 0.587, 0.114], dtype=cp.float32)
    grayscale = cp.dot(frames_gpu, weights)
    return grayscale.astype(cp.uint8)


def simple_threshold_gpu(frames_gpu: 'cp.ndarray', threshold: int) -> 'cp.ndarray':
    """Simple threshold-based conversion using GPU."""
    return (frames_gpu > threshold).astype(cp.uint8)


def floyd_steinberg_dither_gpu(frames_gpu: 'cp.ndarray', threshold: int) -> 'cp.ndarray':
    """
    Floyd-Steinberg dithering using GPU.
    
    Note: True Floyd-Steinberg is inherently sequential, so this implementation
    uses a parallel approximation with random noise to simulate the dithering effect.
    """
    # Add controlled random noise to simulate dithering effect
    noise_scale = 16.0  # Adjust noise intensity
    noise = cp.random.uniform(-noise_scale, noise_scale, frames_gpu.shape).astype(cp.float32)
    
    # Apply noise and threshold
    dithered = frames_gpu.astype(cp.float32) + noise
    
    # Clamp values to valid range
    dithered = cp.clip(dithered, 0, 255)
    
    # Apply threshold
    return (dithered > threshold).astype(cp.uint8)


def ordered_dither_gpu(frames_gpu: 'cp.ndarray', threshold: int, pattern: str) -> 'cp.ndarray':
    """Ordered (Bayer) dithering using GPU."""
    dither_matrix = _get_dither_matrix_gpu(pattern)
    matrix_size = dither_matrix.shape[0]
    
    # Get frame dimensions
    height, width = frames_gpu.shape[1], frames_gpu.shape[2]
    
    # Create coordinate grids for tiling the dither matrix
    y_coords = cp.arange(height)[:, None] % matrix_size
    x_coords = cp.arange(width)[None, :] % matrix_size
    
    # Get dither values for all positions
    dither_values = dither_matrix[y_coords, x_coords]
    
    # Calculate adjusted threshold for each pixel position
    adjusted_threshold = threshold + (dither_values - 0.5) * 64
    
    # Apply threshold to all frames
    frames_float = frames_gpu.astype(cp.float32)
    output_frames = (frames_float > adjusted_threshold[None, :, :]).astype(cp.uint8)
    
    return output_frames


def _get_dither_matrix_gpu(pattern: str) -> 'cp.ndarray':
    """Get dithering matrix for the specified pattern on GPU."""
    matrices = {
        'bayer2x2': cp.array([
            [0, 2],
            [3, 1]
        ], dtype=cp.float32) / 4.0,
        
        'bayer4x4': cp.array([
            [ 0,  8,  2, 10],
            [12,  4, 14,  6],
            [ 3, 11,  1,  9],
            [15,  7, 13,  5]
        ], dtype=cp.float32) / 16.0,
        
        'bayer8x8': cp.array([
            [ 0, 32,  8, 40,  2, 34, 10, 42],
            [48, 16, 56, 24, 50, 18, 58, 26],
            [12, 44,  4, 36, 14, 46,  6, 38],
            [60, 28, 52, 20, 62, 30, 54, 22],
            [ 3, 35, 11, 43,  1, 33,  9, 41],
            [51, 19, 59, 27, 49, 17, 57, 25],
            [15, 47,  7, 39, 13, 45,  5, 37],
            [63, 31, 55, 23, 61, 29, 53, 21]
        ], dtype=cp.float32) / 64.0
    }
    
    if pattern not in matrices:
        raise ValueError(f"Unknown dither pattern: {pattern}")
    
    return matrices[pattern]


# Custom CUDA kernels for advanced dithering (if needed)
def _create_floyd_steinberg_kernel():
    """
    Create a custom CUDA kernel for Floyd-Steinberg dithering.
    
    Note: This is a more complex implementation that could be added
    for true Floyd-Steinberg dithering on GPU, but requires careful
    handling of the sequential nature of the algorithm.
    """
    kernel_code = '''
    extern "C" __global__
    void floyd_steinberg_kernel(float* input, unsigned char* output, 
                               int height, int width, int threshold) {
        int frame_idx = blockIdx.z;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (x >= width || y >= height) return;
        
        int idx = frame_idx * height * width + y * width + x;
        float old_pixel = input[idx];
        float new_pixel = (old_pixel > threshold) ? 255.0f : 0.0f;
        
        output[idx] = (new_pixel > 0) ? 1 : 0;
        
        float error = old_pixel - new_pixel;
        
        // Distribute error (simplified for parallel execution)
        // Note: True Floyd-Steinberg requires sequential processing
        if (x + 1 < width) {
            atomicAdd(&input[idx + 1], error * 7.0f / 16.0f);
        }
        if (y + 1 < height) {
            if (x > 0) {
                atomicAdd(&input[idx + width - 1], error * 3.0f / 16.0f);
            }
            atomicAdd(&input[idx + width], error * 5.0f / 16.0f);
            if (x + 1 < width) {
                atomicAdd(&input[idx + width + 1], error * 1.0f / 16.0f);
            }
        }
    }
    '''
    return kernel_code


# Utility functions for performance optimization
def optimize_gpu_memory_usage(frames_shape: tuple, device_id: int = 0) -> Dict[str, float]:
    """
    Analyze GPU memory usage for 1-bit conversion operations.
    
    Args:
        frames_shape: Shape of input frames
        device_id: CUDA device ID
        
    Returns:
        Dictionary with memory usage information
    """
    if not CUDA_AVAILABLE:
        return {'error': 'CUDA not available'}
    
    with cp.cuda.Device(device_id):
        # Get device memory info
        mempool = cp.get_default_memory_pool()
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        
        # Calculate memory requirements
        frame_bytes = np.prod(frames_shape) * 1  # uint8
        float_bytes = np.prod(frames_shape) * 4  # float32 for processing
        
        # Estimate total memory needed (input + working + output)
        total_needed = frame_bytes * 2 + float_bytes  # Conservative estimate
        
        return {
            'total_gpu_memory_mb': total_bytes / (1024 * 1024),
            'free_gpu_memory_mb': free_bytes / (1024 * 1024),
            'estimated_needed_mb': total_needed / (1024 * 1024),
            'memory_utilization_percent': (total_needed / free_bytes) * 100,
            'can_fit_in_memory': total_needed < free_bytes * 0.8  # 80% safety margin
        }
