"""
Video frame rate conversion component using intelligent temporal downsampling.

This component provides video frame rate conversion with configurable algorithms,
supporting both CPU and GPU acceleration for optimal performance.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
import logging

from . import ChainComponent, VideoData, ProcessingError, ProgressReporter, MemoryManager, LogManager


class VideoReframer(ChainComponent):
    """Component for frame rate conversion using temporal processing."""
    
    def __init__(self,
                 target_fps: float,
                 technique: str = 'intelligent',
                 use_gpu: bool = False,
                 gpu_device: int = 0,
                 variance_weight: float = 0.7,
                 difference_weight: float = 0.3):
        """
        Initialize the video reframer.
        
        Args:
            target_fps: Target frame rate for output video
            technique: Resampling technique ('simple' or 'intelligent')
            use_gpu: Whether to use GPU acceleration
            gpu_device: GPU device ID to use
            variance_weight: Weight for frame variance in intelligent selection (0-1)
            difference_weight: Weight for temporal difference in intelligent selection (0-1)
        """
        super().__init__("VideoReframer")
        self.target_fps = target_fps
        self.technique = technique
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        self.variance_weight = variance_weight
        self.difference_weight = difference_weight
        
        # Validate parameters
        if target_fps <= 0:
            raise ValueError(f"Target FPS must be positive, got {target_fps}")
        
        if technique not in ['simple', 'intelligent']:
            raise ValueError(f"Unknown technique: {technique}. Use 'simple' or 'intelligent'")
        
        if not (0 <= variance_weight <= 1) or not (0 <= difference_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")
        
        if abs((variance_weight + difference_weight) - 1.0) > 0.01:
            self.logger.warning(f"Weights sum to {variance_weight + difference_weight:.3f}, not 1.0")
        
        # Initialize resampling function
        self.resample_function = self._get_resample_function()
        
        LogManager.log_info(self.name, f"Initialized: target_fps={target_fps}, technique={technique}, GPU={use_gpu}")
    
    def _get_resample_function(self):
        """Get the appropriate resampling function based on settings."""
        try:
            if self.use_gpu:
                # Try to import GPU functions
                try:
                    from ..cuda.frame_resampling import frame_resample_gpu
                    return frame_resample_gpu
                except ImportError:
                    self.logger.warning("CUDA not available, falling back to CPU")
                    LogManager.log_warning(self.name, "CUDA not available, falling back to CPU")
                    self.use_gpu = False
            
            # Use CPU functions
            from ..cpu.frame_resampling import frame_resample_cpu
            return frame_resample_cpu
            
        except ImportError as e:
            raise ProcessingError(
                f"Could not import resampling functions: {e}",
                component=self.name
            )
    
    def _validate_input(self, data: VideoData):
        """Validate input data for reframing."""
        if data.frames is None or len(data.frames) == 0:
            raise ProcessingError(
                "No frames to process",
                component=self.name
            )
        
        if data.frame_rate <= 0:
            raise ProcessingError(
                f"Invalid source frame rate: {data.frame_rate}",
                component=self.name
            )
        
        if self.target_fps > data.frame_rate:
            raise ProcessingError(
                f"Target FPS ({self.target_fps}) cannot be higher than source FPS ({data.frame_rate})",
                component=self.name,
                details={
                    'source_fps': data.frame_rate,
                    'target_fps': self.target_fps
                }
            )
    
    def process(self, data: VideoData) -> VideoData:
        """
        Process video data with frame rate conversion.
        
        Args:
            data: Input video data
            
        Returns:
            VideoData with converted frame rate
        """
        self._validate_input(data)
        
        # Log processing start
        LogManager.log_info(
            self.name, 
            f"Starting reframe operation: {data.frame_rate:.2f} -> {self.target_fps:.2f} FPS, {data.frame_count} frames"
        )
        
        try:
            # Check if reframing is needed
            if abs(data.frame_rate - self.target_fps) < 0.01:
                self.logger.info(f"Source FPS ({data.frame_rate:.2f}) matches target FPS ({self.target_fps:.2f}), skipping reframe")
                LogManager.log_info(self.name, f"Frame rates match, skipping reframe operation")
                return data
            
            # Perform resampling
            resampled_frames, selected_indices, statistics = self.resample_function(
                data.frames,
                data.frame_rate,
                self.target_fps,
                self.technique
            )
            
            # Log statistics
            LogManager.log_info(self.name, f"Reframe statistics: {statistics}")
            self.logger.info(f"Resampled from {statistics['original_frame_count']} to {statistics['output_frame_count']} frames")
            self.logger.info(f"Frame drop percentage: {statistics['frame_drop_percentage']:.1f}%")
            
            # Create output VideoData
            output_data = VideoData(
                frames=resampled_frames,
                frame_rate=self.target_fps,
                resolution=data.resolution,
                color_mode=data.color_mode,
                metadata=data.metadata.copy()
            )
            
            # Update metadata
            output_data.metadata.update({
                'reframe_statistics': statistics,
                'reframe_technique': self.technique,
                'reframe_weights': {
                    'variance': self.variance_weight,
                    'difference': self.difference_weight
                },
                'selected_frame_indices': selected_indices.tolist() if hasattr(selected_indices, 'tolist') else list(selected_indices),
                'processing_device': 'GPU' if self.use_gpu else 'CPU'
            })
            
            LogManager.log_info(self.name, f"Reframe completed: {data.frame_count} -> {output_data.frame_count} frames")
            
            return output_data
            
        except Exception as e:
            LogManager.log_error(self.name, f"Reframe operation failed: {str(e)}", e)
            raise ProcessingError(
                f"Reframe operation failed: {str(e)}",
                component=self.name,
                details={
                    'source_fps': data.frame_rate,
                    'target_fps': self.target_fps,
                    'technique': self.technique
                }
            )


# Convenience function for quick video reframing
def reframe_video(data: VideoData,
                 target_fps: float,
                 technique: str = 'intelligent',
                 use_gpu: bool = False,
                 **kwargs) -> VideoData:
    """
    Convenience function for video frame rate conversion.
    
    Args:
        data: Input video data
        target_fps: Target frame rate
        technique: Resampling technique ('simple' or 'intelligent')
        use_gpu: Whether to use GPU acceleration
        **kwargs: Additional parameters for VideoReframer
        
    Returns:
        VideoData with converted frame rate
    """
    reframer = VideoReframer(
        target_fps=target_fps,
        technique=technique,
        use_gpu=use_gpu,
        **kwargs
    )
    return reframer.process(data)


__all__ = ['VideoReframer', 'reframe_video']
