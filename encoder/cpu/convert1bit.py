"""
CPU-based 1-bit video conversion with dithering support.

This module provides CPU-optimized implementations for converting RGB/Grayscale
video to 1-bit with various dithering algorithms using Numba JIT compilation.
"""

import numpy as np
from typing import Optional, Literal, Dict
import time
from numba import jit, prange


@jit(nopython=True, parallel=True)
def rgb_to_grayscale_cpu(frames: np.ndarray) -> np.ndarray:
    """Convert RGB frames to grayscale using CPU."""
    output = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2]), dtype=np.uint8)
    for i in prange(frames.shape[0]):
        for y in prange(frames.shape[1]):
            for x in prange(frames.shape[2]):
                # Standard RGB to grayscale weights
                gray = int(0.299 * frames[i, y, x, 0] + 
                          0.587 * frames[i, y, x, 1] + 
                          0.114 * frames[i, y, x, 2])
                output[i, y, x] = min(255, max(0, gray))
    return output


@jit(nopython=True, parallel=True)
def simple_threshold_cpu(frames: np.ndarray, threshold: int) -> np.ndarray:
    """Simple threshold-based conversion using CPU."""
    output = np.zeros_like(frames, dtype=np.uint8)
    for i in prange(frames.shape[0]):
        for y in prange(frames.shape[1]):
            for x in prange(frames.shape[2]):
                output[i, y, x] = 1 if frames[i, y, x] > threshold else 0
    return output


@jit(nopython=True)
def floyd_steinberg_dither_cpu(frames: np.ndarray, threshold: int) -> np.ndarray:
    """Floyd-Steinberg dithering using CPU."""
    output_frames = np.zeros_like(frames, dtype=np.uint8)
    
    for frame_idx in range(frames.shape[0]):
        frame = frames[frame_idx].astype(np.float32)
        height, width = frame.shape
        
        # Process each pixel
        for y in range(height):
            for x in range(width):
                old_pixel = frame[y, x]
                new_pixel = 255.0 if old_pixel > threshold else 0.0
                output_frames[frame_idx, y, x] = 1 if new_pixel > 0 else 0
                
                # Calculate quantization error
                quant_error = old_pixel - new_pixel
                
                # Distribute error to neighboring pixels
                if x + 1 < width:
                    frame[y, x + 1] += quant_error * 7.0 / 16.0
                if y + 1 < height:
                    if x > 0:
                        frame[y + 1, x - 1] += quant_error * 3.0 / 16.0
                    frame[y + 1, x] += quant_error * 5.0 / 16.0
                    if x + 1 < width:
                        frame[y + 1, x + 1] += quant_error * 1.0 / 16.0
    
    return output_frames


def _get_dither_matrix(pattern: str) -> np.ndarray:
    """Get dithering matrix for the specified pattern."""
    matrices = {
        'bayer2x2': np.array([
            [0, 2],
            [3, 1]
        ], dtype=np.float32) / 4.0,
        
        'bayer4x4': np.array([
            [ 0,  8,  2, 10],
            [12,  4, 14,  6],
            [ 3, 11,  1,  9],
            [15,  7, 13,  5]
        ], dtype=np.float32) / 16.0,
        
        'bayer8x8': np.array([
            [ 0, 32,  8, 40,  2, 34, 10, 42],
            [48, 16, 56, 24, 50, 18, 58, 26],
            [12, 44,  4, 36, 14, 46,  6, 38],
            [60, 28, 52, 20, 62, 30, 54, 22],
            [ 3, 35, 11, 43,  1, 33,  9, 41],
            [51, 19, 59, 27, 49, 17, 57, 25],
            [15, 47,  7, 39, 13, 45,  5, 37],
            [63, 31, 55, 23, 61, 29, 53, 21]
        ], dtype=np.float32) / 64.0
    }
    
    if pattern not in matrices:
        raise ValueError(f"Unknown dither pattern: {pattern}")
    
    return matrices[pattern]


@jit(nopython=True, parallel=True)
def ordered_dither_cpu(frames: np.ndarray, threshold: int, dither_matrix: np.ndarray) -> np.ndarray:
    """Ordered (Bayer) dithering using CPU."""
    output_frames = np.zeros_like(frames, dtype=np.uint8)
    matrix_size = dither_matrix.shape[0]
    
    for frame_idx in prange(frames.shape[0]):
        frame = frames[frame_idx].astype(np.float32)
        height, width = frame.shape
        
        for y in prange(height):
            for x in prange(width):
                # Get dither threshold for this pixel position
                dither_value = dither_matrix[y % matrix_size, x % matrix_size]
                adjusted_threshold = threshold + (dither_value - 0.5) * 64
                
                # Apply threshold
                output_frames[frame_idx, y, x] = 1 if frame[y, x] > adjusted_threshold else 0
    
    return output_frames


def convert_1bit_cpu(frames: np.ndarray, 
                     threshold: int = 128,
                     dither_method: Optional[Literal['none', 'floyd-steinberg', 'ordered']] = 'none',
                     dither_pattern: Literal['bayer2x2', 'bayer4x4', 'bayer8x8'] = 'bayer4x4') -> np.ndarray:
    """
    Convert video frames to 1-bit using CPU processing.
    
    Args:
        frames: Input video frames (N, H, W) or (N, H, W, C)
        threshold: Threshold value for binary conversion (0-255)
        dither_method: Dithering method ('none', 'floyd-steinberg', 'ordered')
        dither_pattern: Pattern for ordered dithering
        
    Returns:
        1-bit video frames as uint8 array (0 or 1 values)
    """
    # Convert to grayscale if needed
    if len(frames.shape) == 4 and frames.shape[-1] == 3:
        grayscale_frames = rgb_to_grayscale_cpu(frames)
    elif len(frames.shape) == 4 and frames.shape[-1] == 1:
        grayscale_frames = frames.squeeze(-1)
    else:
        grayscale_frames = frames
    
    # Apply conversion method
    if dither_method == 'none':
        return simple_threshold_cpu(grayscale_frames, threshold)
    elif dither_method == 'floyd-steinberg':
        return floyd_steinberg_dither_cpu(grayscale_frames, threshold)
    elif dither_method == 'ordered':
        dither_matrix = _get_dither_matrix(dither_pattern)
        return ordered_dither_cpu(grayscale_frames, threshold, dither_matrix)
    else:
        raise ValueError(f"Unknown dither method: {dither_method}")
