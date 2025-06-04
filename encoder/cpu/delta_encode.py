"""
CPU-optimized delta encoding with RLE compression.

This module provides CPU-optimized implementations for delta encoding and
RLE compression using Numba JIT compilation for parallel processing.
"""

import numpy as np
from typing import Optional, Tuple
from numba import jit, prange
import math


@jit(nopython=True, parallel=True)
def compute_row_delta(prev_row: np.ndarray, curr_row: np.ndarray) -> np.ndarray:
    """Compute delta between two rows of 1-bit pixels."""
    delta = np.zeros_like(curr_row)
    for x in prange(curr_row.shape[0]):
        # XOR to find changed pixels
        delta[x] = curr_row[x] ^ prev_row[x]
    return delta


@jit(nopython=True, parallel=True)
def compute_frame_deltas_cpu(frames: np.ndarray, workers: int = 4) -> np.ndarray:
    """
    Compute frame deltas using parallel row processing.
    
    Args:
        frames: Input 1-bit video frames (N, H, W)
        workers: Number of worker threads for parallel processing
        
    Returns:
        Frame deltas with same shape as input (N, H, W)
    """
    frame_count, height, width = frames.shape
    deltas = np.zeros_like(frames)
    
    # First frame is copied as-is since it has no previous frame
    deltas[0] = frames[0]
    
    # Process remaining frames in parallel by rows
    for frame_idx in range(1, frame_count):
        prev_frame = frames[frame_idx - 1]
        curr_frame = frames[frame_idx]
        
        # Process each row in parallel
        for y in prange(height):
            deltas[frame_idx, y] = compute_row_delta(prev_frame[y], curr_frame[y])
    
    return deltas


@jit(nopython=True)
def compress_row_rle(row: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Compress a row using run-length encoding.
    
    Returns:
        Tuple of (compressed data, compressed length)
    """
    # Pre-allocate max possible size (alternating runs)
    max_runs = row.shape[0]
    compressed = np.zeros(max_runs * 2, dtype=np.uint8)
    
    pos = 0  # Position in compressed output
    count = 1  # Current run length
    curr_val = row[0]  # Current value being counted
    
    # Process remaining pixels
    for i in range(1, row.shape[0]):
        if row[i] == curr_val and count < 255:
            count += 1
        else:
            # Store run
            compressed[pos] = count
            compressed[pos + 1] = curr_val
            pos += 2
            # Start new run
            curr_val = row[i]
            count = 1
    
    # Store final run
    compressed[pos] = count
    compressed[pos + 1] = curr_val
    pos += 2
    
    return compressed[:pos], pos


@jit(nopython=True, parallel=True)
def compress_deltas_rle_cpu(deltas: np.ndarray, workers: int = 4) -> np.ndarray:
    """
    Compress frame deltas using RLE with parallel row processing.
    
    Args:
        deltas: Input delta frames (N, H, W)
        workers: Number of worker threads
        
    Returns:
        Compressed frames with shape (N, H, W') where W' is the compressed width
    """
    frame_count, height, width = deltas.shape
    
    # Pre-allocate with worst-case size (no compression)
    # We'll trim this later after calculating actual compressed sizes
    compressed = np.zeros((frame_count, height, width * 2), dtype=np.uint8)
    compressed_sizes = np.zeros((frame_count, height), dtype=np.int32)
    
    # Process each frame's rows in parallel
    for frame_idx in prange(frame_count):
        for y in prange(height):
            row_data, row_size = compress_row_rle(deltas[frame_idx, y])
            compressed[frame_idx, y, :row_size] = row_data
            compressed_sizes[frame_idx, y] = row_size
    
    # Find maximum compressed row size to determine final array shape
    max_compressed_size = np.max(compressed_sizes)
    
    # Return array trimmed to maximum compressed size
    return compressed[:, :, :max_compressed_size]
