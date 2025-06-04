"""
GPU-based frame resampling for video frame rate conversion using CuPy.

This module provides CUDA-accelerated frame rate conversion with intelligent frame selection
algorithms for optimal GPU performance.
"""

from . import CUDA_AVAILABLE, cp

if not CUDA_AVAILABLE:
    def frame_resample_gpu(*args, **kwargs):
        raise RuntimeError("CUDA not available. No CUDA devices found or CuPy not installed.")
    
    def simple_downsample_gpu(*args, **kwargs):
        raise RuntimeError("CUDA not available. No CUDA devices found or CuPy not installed.")
    
    def intelligent_downsample_gpu(*args, **kwargs):
        raise RuntimeError("CUDA not available. No CUDA devices found or CuPy not installed.")
        
else:
    import numpy as np
    from typing import Tuple, Optional
    
    def calculate_frame_variance_gpu(frames: cp.ndarray) -> cp.ndarray:
        """Calculate variance for all frames on GPU."""
        # Calculate mean for each frame
        if len(frames.shape) == 4:
            # Color frames (N, H, W, C)
            frame_means = cp.mean(frames, axis=(1, 2, 3))
            frame_vars = cp.mean((frames - frame_means[:, cp.newaxis, cp.newaxis, cp.newaxis]) ** 2, axis=(1, 2, 3))
        else:
            # Grayscale frames (N, H, W)
            frame_means = cp.mean(frames, axis=(1, 2))
            frame_vars = cp.mean((frames - frame_means[:, cp.newaxis, cp.newaxis]) ** 2, axis=(1, 2))
        
        return frame_vars
    
    def calculate_temporal_differences_gpu(frames: cp.ndarray) -> cp.ndarray:
        """Calculate temporal differences between consecutive frames on GPU."""
        if frames.shape[0] < 2:
            return cp.zeros(frames.shape[0], dtype=cp.float32)
        
        # Calculate differences between consecutive frames
        frame_diffs = cp.zeros(frames.shape[0], dtype=cp.float32)
        
        # Convert to float32 for calculations
        frames_float = frames.astype(cp.float32)
        
        for i in range(1, frames.shape[0]):
            diff = cp.mean(cp.abs(frames_float[i] - frames_float[i-1]))
            frame_diffs[i] = diff
        
        return frame_diffs
    
    def simple_downsample_gpu(frames: cp.ndarray, source_fps: float, target_fps: float) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Simple frame dropping algorithm for GPU.
        
        Args:
            frames: Input frames array on GPU (N, H, W, C) or (N, H, W)
            source_fps: Original frame rate
            target_fps: Target frame rate
            
        Returns:
            Tuple of (selected_frames, selected_indices)
        """
        if target_fps >= source_fps:
            # No downsampling needed
            indices = cp.arange(frames.shape[0], dtype=cp.int32)
            return frames, indices
        
        frame_count = frames.shape[0]
        
        # Calculate frame selection ratio
        ratio = source_fps / target_fps
        
        # Calculate number of output frames
        output_count = int(frame_count / ratio)
        
        # Calculate which frames to select using vectorized operations
        output_indices = cp.arange(output_count, dtype=cp.float32)
        source_indices = (output_indices * ratio).astype(cp.int32)
        
        # Clamp indices to valid range
        source_indices = cp.clip(source_indices, 0, frame_count - 1)
        
        # Select frames using advanced indexing
        selected_frames = frames[source_indices]
        
        return selected_frames, source_indices
    
    def intelligent_downsample_gpu(frames: cp.ndarray, source_fps: float, target_fps: float,
                                  variance_weight: float = 0.7, difference_weight: float = 0.3) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Intelligent frame selection algorithm for GPU using variance and temporal difference.
        
        Args:
            frames: Input frames array on GPU (N, H, W, C) or (N, H, W)
            source_fps: Original frame rate
            target_fps: Target frame rate
            variance_weight: Weight for frame variance in selection (0-1)
            difference_weight: Weight for temporal difference in selection (0-1)
            
        Returns:
            Tuple of (selected_frames, selected_indices)
        """
        if target_fps >= source_fps:
            # No downsampling needed
            indices = cp.arange(frames.shape[0], dtype=cp.int32)
            return frames, indices
        
        frame_count = frames.shape[0]
        ratio = source_fps / target_fps
        output_count = int(frame_count / ratio)
        
        # Calculate frame variances (quality metric) on GPU
        variances = calculate_frame_variance_gpu(frames)
        
        # Calculate temporal differences (motion metric) on GPU
        temporal_diffs = calculate_temporal_differences_gpu(frames)
        
        # Normalize metrics
        var_max = cp.max(variances) if cp.max(variances) > 0 else 1.0
        diff_max = cp.max(temporal_diffs) if cp.max(temporal_diffs) > 0 else 1.0
        
        variances = variances / var_max
        temporal_diffs = temporal_diffs / diff_max
        
        # Calculate combined scores
        scores = variance_weight * variances + difference_weight * temporal_diffs
        
        # Select best frames in each window
        selected_indices = []
        
        for window_idx in range(output_count):
            # Define window boundaries
            window_start = int(window_idx * ratio)
            window_end = min(int((window_idx + 1) * ratio), frame_count)
            
            if window_start >= frame_count:
                break
                
            if window_end <= window_start:
                window_end = window_start + 1
            
            # Extract window scores
            window_scores = scores[window_start:window_end]
            
            # Find best frame index in window
            local_best_idx = cp.argmax(window_scores)
            global_best_idx = window_start + local_best_idx
            
            selected_indices.append(int(global_best_idx))
        
        # Convert to CuPy array
        selected_indices = cp.array(selected_indices, dtype=cp.int32)
        
        # Select frames using advanced indexing
        selected_frames = frames[selected_indices]
        
        return selected_frames, selected_indices
    
    def frame_resample_gpu(frames, source_fps: float, target_fps: float, 
                          technique: str = 'intelligent') -> Tuple:
        """
        GPU-based frame resampling with selectable techniques.
        
        Args:
            frames: Input frames array (numpy or cupy array) (N, H, W, C) or (N, H, W)
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
        
        # Ensure frames are on GPU
        if not isinstance(frames, cp.ndarray):
            frames_gpu = cp.asarray(frames)
        else:
            frames_gpu = frames
        
        original_count = frames_gpu.shape[0]
        
        if technique == 'simple':
            selected_frames, selected_indices = simple_downsample_gpu(frames_gpu, source_fps, target_fps)
        elif technique == 'intelligent':
            selected_frames, selected_indices = intelligent_downsample_gpu(frames_gpu, source_fps, target_fps)
        else:
            raise ValueError(f"Unknown technique: {technique}. Use 'simple' or 'intelligent'")
        
        # Convert results back to NumPy for compatibility with VideoData
        if hasattr(selected_frames, 'get'):
            selected_frames = selected_frames.get()
        if hasattr(selected_indices, 'get'):
            selected_indices = selected_indices.get()
        
        # Calculate statistics
        statistics = {
            'original_frame_count': original_count,
            'output_frame_count': len(selected_indices),
            'compression_ratio': len(selected_indices) / original_count if original_count > 0 else 0,
            'source_fps': source_fps,
            'target_fps': target_fps,
            'technique': technique,
            'frame_drop_percentage': (1 - len(selected_indices) / original_count) * 100 if original_count > 0 else 0,
            'processing_device': 'GPU'
        }
        
        return selected_frames, selected_indices, statistics


__all__ = ['frame_resample_gpu', 'simple_downsample_gpu', 'intelligent_downsample_gpu'] if CUDA_AVAILABLE else []
