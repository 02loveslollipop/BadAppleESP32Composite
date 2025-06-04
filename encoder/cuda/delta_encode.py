import cupy as cp
import numpy as np
from numba import cuda

def compute_frame_deltas_gpu(frames):
    """
    Compute frame deltas using GPU acceleration.
    Each row is processed in parallel on the GPU.
    
    Args:
        frames (cp.ndarray): Input frames as a 3D array (frames, height, width) of 1-bit values
    
    Returns:
        cp.ndarray: Delta-encoded frames
    """
    if not isinstance(frames, cp.ndarray):
        frames = cp.asarray(frames)
    
    # Ensure input is properly shaped and contains 1-bit values
    assert frames.dtype == cp.uint8
    assert cp.all((frames == 0) | (frames == 1))
    
    # Create output array
    deltas = cp.zeros_like(frames)
    deltas[0] = frames[0]  # First frame is unchanged
    
    # Compute deltas between consecutive frames
    deltas[1:] = frames[1:] ^ frames[:-1]
    
    return deltas

def compress_deltas_rle_gpu(delta_frames):
    """
    Compress delta-encoded frames using RLE compression on GPU.
    
    Args:
        delta_frames (cp.ndarray): Delta-encoded frames
        
    Returns:
        list: List of compressed frames, each containing run lengths
    """
    if not isinstance(delta_frames, cp.ndarray):
        delta_frames = cp.asarray(delta_frames)
    
    compressed_frames = []
    
    for frame in delta_frames:
        # Reshape to 1D for easier RLE compression
        flat_frame = frame.ravel()
        
        # Find indices where values change
        value_changes = cp.diff(cp.concatenate(([2], flat_frame, [2]))) != 0
        change_indices = cp.where(value_changes)[0]
        
        # Calculate run lengths
        run_lengths = cp.diff(change_indices)
        
        # Get values for each run
        values = flat_frame[change_indices[:-1]]
        
        # Combine into compressed format and convert to host
        compressed = cp.stack((values, run_lengths), axis=1).get()
        compressed_frames.append(compressed)
    
    return compressed_frames
