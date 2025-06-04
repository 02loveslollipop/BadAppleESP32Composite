"""
CPU-based frame resampling for video frame rate conversion.

This module provides frame rate conversion with intelligent frame selection algorithms
using Numba JIT compilation for optimal CPU performance.
"""

import numpy as np
from numba import jit, prange
import math
from typing import List, Tuple, Optional


@jit(nopython=True)
def calculate_frame_variance(frame: np.ndarray) -> float:
    """Calculate variance of a frame for quality assessment."""
    if len(frame.shape) == 3:
        # Color frame - calculate variance across all channels
        mean_val = np.mean(frame)
        variance = np.mean((frame - mean_val) ** 2)
    else:
        # Grayscale frame
        mean_val = np.mean(frame)
        variance = np.mean((frame - mean_val) ** 2)
    return variance


@jit(nopython=True)
def calculate_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Calculate difference between two frames."""
    if frame1.shape != frame2.shape:
        return float('inf')  # Incompatible frames
    
    diff = np.mean(np.abs(frame1.astype(np.float32) - frame2.astype(np.float32)))
    return diff


@jit(nopython=True, parallel=True)
def simple_downsample_cpu(frames: np.ndarray, source_fps: float, target_fps: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple frame dropping algorithm for CPU.
    
    Args:
        frames: Input frames array (N, H, W, C) or (N, H, W)
        source_fps: Original frame rate
        target_fps: Target frame rate
        
    Returns:
        Tuple of (selected_frames, selected_indices)
    """
    if target_fps >= source_fps:
        # No downsampling needed
        indices = np.arange(frames.shape[0], dtype=np.int32)
        return frames, indices
    
    frame_count = frames.shape[0]
    
    # Calculate frame selection ratio
    ratio = source_fps / target_fps
    
    # Calculate number of output frames
    output_count = int(frame_count / ratio)
    
    # Calculate which frames to select
    selected_indices = np.zeros(output_count, dtype=np.int32)
    
    for i in prange(output_count):
        # Calculate source frame index
        source_idx = int(i * ratio)
        if source_idx >= frame_count:
            source_idx = frame_count - 1
        selected_indices[i] = source_idx
    
    # Extract selected frames
    if len(frames.shape) == 4:
        # Color frames
        selected_frames = np.zeros((output_count, frames.shape[1], frames.shape[2], frames.shape[3]), dtype=frames.dtype)
        for i in prange(output_count):
            selected_frames[i] = frames[selected_indices[i]]
    else:
        # Grayscale frames
        selected_frames = np.zeros((output_count, frames.shape[1], frames.shape[2]), dtype=frames.dtype)
        for i in prange(output_count):
            selected_frames[i] = frames[selected_indices[i]]
    
    return selected_frames, selected_indices


def intelligent_downsample_cpu(frames: np.ndarray, source_fps: float, target_fps: float, 
                              variance_weight: float = 0.7, difference_weight: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Intelligent frame selection algorithm for CPU using variance and temporal difference.
    
    Args:
        frames: Input frames array (N, H, W, C) or (N, H, W)
        source_fps: Original frame rate
        target_fps: Target frame rate
        variance_weight: Weight for frame variance in selection (0-1)
        difference_weight: Weight for temporal difference in selection (0-1)
        
    Returns:
        Tuple of (selected_frames, selected_indices)
    """
    if target_fps >= source_fps:
        # No downsampling needed
        indices = np.arange(frames.shape[0], dtype=np.int32)
        return frames, indices
    
    frame_count = frames.shape[0]
    ratio = source_fps / target_fps
    output_count = int(frame_count / ratio)
    
    # Calculate frame variances (quality metric)
    variances = np.zeros(frame_count, dtype=np.float32)
    for i in range(frame_count):
        variances[i] = calculate_frame_variance(frames[i])
    
    # Calculate temporal differences (motion metric)
    temporal_diffs = np.zeros(frame_count, dtype=np.float32)
    for i in range(1, frame_count):
        temporal_diffs[i] = calculate_frame_difference(frames[i-1], frames[i])
    
    # Normalize metrics
    var_max = np.max(variances) if np.max(variances) > 0 else 1.0
    diff_max = np.max(temporal_diffs) if np.max(temporal_diffs) > 0 else 1.0
    
    variances = variances / var_max
    temporal_diffs = temporal_diffs / diff_max
    
    # Calculate selection windows
    selected_indices = []
    
    for window_idx in range(output_count):
        # Define window boundaries
        window_start = int(window_idx * ratio)
        window_end = min(int((window_idx + 1) * ratio), frame_count)
        
        if window_start >= frame_count:
            break
            
        if window_end <= window_start:
            window_end = window_start + 1
        
        # Find best frame in window using combined metric
        best_score = -1.0
        best_idx = window_start
        
        for idx in range(window_start, window_end):
            # Combined score: high variance (detail) + high temporal difference (motion)
            score = variance_weight * variances[idx] + difference_weight * temporal_diffs[idx]
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        selected_indices.append(best_idx)
    
    # Convert to numpy array
    selected_indices = np.array(selected_indices, dtype=np.int32)
    
    # Extract selected frames
    if len(frames.shape) == 4:
        # Color frames
        selected_frames = frames[selected_indices]
    else:
        # Grayscale frames
        selected_frames = frames[selected_indices]
    
    return selected_frames, selected_indices


def frame_resample_cpu(frames: np.ndarray, source_fps: float, target_fps: float, 
                      technique: str = 'intelligent') -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    CPU-based frame resampling with selectable techniques.
    
    Args:
        frames: Input frames array (N, H, W, C) or (N, H, W)
        source_fps: Original frame rate
        target_fps: Target frame rate
        technique: Resampling technique ('simple' or 'intelligent')
        
    Returns:
        Tuple of (resampled_frames, selected_indices, statistics)
    """
    if target_fps > source_fps:
        raise ValueError(f"Target FPS ({target_fps}) cannot be higher than source FPS ({source_fps})")
    
    if target_fps <= 0 or source_fps <= 0:
        raise ValueError("Frame rates must be positive")
    
    original_count = frames.shape[0]
    
    if technique == 'simple':
        selected_frames, selected_indices = simple_downsample_cpu(frames, source_fps, target_fps)
    elif technique == 'intelligent':
        selected_frames, selected_indices = intelligent_downsample_cpu(frames, source_fps, target_fps)
    else:
        raise ValueError(f"Unknown technique: {technique}. Use 'simple' or 'intelligent'")
    
    # Calculate statistics
    statistics = {
        'original_frame_count': original_count,
        'output_frame_count': len(selected_indices),
        'compression_ratio': len(selected_indices) / original_count if original_count > 0 else 0,
        'source_fps': source_fps,
        'target_fps': target_fps,
        'technique': technique,
        'frame_drop_percentage': (1 - len(selected_indices) / original_count) * 100 if original_count > 0 else 0
    }
    
    return selected_frames, selected_indices, statistics


__all__ = ['frame_resample_cpu', 'simple_downsample_cpu', 'intelligent_downsample_cpu']
