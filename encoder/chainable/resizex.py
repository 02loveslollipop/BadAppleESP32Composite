"""
Video resizing component using Lanczos interpolation.

This component provides video resolution scaling with configurable Lanczos kernel sizes,
supporting both CPU and GPU acceleration.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
import logging

from . import ChainComponent, VideoData, ProcessingError, ProgressReporter, MemoryManager, LogManager
from ..video_scheduler import VideoResizeTaskScheduler, FrameData


class VideoResizer(ChainComponent):
    """Component for resizing video frames using Lanczos interpolation."""
    
    def __init__(self,
        target_resolution: Tuple[int, int], lanczos_kernel: int = 4, use_gpu: bool = False,  gpu_device: int = 0, workers: int = 4, execution_mode: str = 'threading', batch_size: int = 10):

        super().__init__("VideoResizer")
        self.target_resolution = target_resolution
        self.lanczos_kernel = lanczos_kernel
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        self.workers = workers
        self.execution_mode = execution_mode
        self.batch_size = batch_size
        
        # Validate parameters
        if lanczos_kernel < 2:
            raise ValueError(f"Lanczos kernel size must be at least 2, got {lanczos_kernel}")
        
        if target_resolution[0] <= 0 or target_resolution[1] <= 0:
            raise ValueError(f"Invalid target resolution: {target_resolution}")
        
        # Initialize the task scheduler with proper resize function
        resize_func = self._get_resize_function()
        
        self.scheduler = VideoResizeTaskScheduler(
            resize_function=resize_func,
            target_width=target_resolution[0],
            target_height=target_resolution[1],
            max_workers=workers,
            use_cuda=use_gpu,
            device_id=gpu_device,
            execution_mode=execution_mode,
            batch_size=batch_size,
            kernel_size=lanczos_kernel
        )
    
    def _get_resize_function(self):
        """Get the appropriate resize function based on settings."""
        try:
            if self.use_gpu:
                # Try to import GPU functions
                try:
                    from ..cuda.lanczos_resize import lanczos4_resize_gpu, lanczos_resize_gpu
                    if self.lanczos_kernel == 4:
                        return lanczos4_resize_gpu
                    else:
                        return lanczos_resize_gpu
                except ImportError:
                    self.logger.warning("CUDA not available, falling back to CPU")
                    self.use_gpu = False
            
            # Use CPU functions
            from ..cpu.lanczos_resize import lanczos4_resize_cpu, lanczos_resize_cpu
            if self.lanczos_kernel == 4:
                return lanczos4_resize_cpu
            else:
                return lanczos_resize_cpu
        except ImportError as e:
            raise ProcessingError(
                f"Could not import resize functions: {e}",
                component=self.name
            )
    
    def _validate_input(self, data: VideoData):
        """Validate input data for resizing."""
        super()._validate_input(data)
        
        if data.frames.size == 0:
            raise ProcessingError(
                "Input video has no frames",
                component=self.name
            )
        
        # Check if resize is actually needed
        current_resolution = (data.width, data.height)
        if current_resolution == self.target_resolution:
            self.logger.warning(
                f"Input resolution {current_resolution} already matches target {self.target_resolution}"
            )
    
    def process(self, data: VideoData) -> VideoData:
        """
        Resize video frames using Lanczos interpolation.
        
        Args:
            data: Input VideoData
            
        Returns:
            VideoData with resized frames
        """
        current_resolution = (data.width, data.height)
        self.logger.info(
            f"Resizing from {current_resolution} to {self.target_resolution} "
            f"using Lanczos-{self.lanczos_kernel} ({'GPU' if self.use_gpu else 'CPU'})"
        )
        LogManager.log_info(
            self.name, 
            f"Starting resize operation: {current_resolution} -> {self.target_resolution}, "
            f"Lanczos-{self.lanczos_kernel}, {'GPU' if self.use_gpu else 'CPU'}, "
            f"{data.frame_count} frames"
        )
          # Check if resize is needed
        if current_resolution == self.target_resolution:
            self.logger.info("No resize needed, returning original data")
            return data

        # Check memory usage and determine processing strategy
        memory_info = MemoryManager.estimate_memory_usage(data)
        self.logger.info(f"Estimated memory usage: {memory_info['total_mb']:.1f} MB")
        LogManager.log_info(self.name, f"Memory analysis: {memory_info['total_mb']:.1f} MB for {data.frame_count} frames")

        if MemoryManager.should_use_batch_processing(data):            
            optimal_batch = MemoryManager.calculate_optimal_batch_size(data)
            self.logger.info(f"Using batch processing with batch size: {optimal_batch}")
            LogManager.log_info(self.name, f"Using batch processing strategy: batch_size={optimal_batch}")
            return self._process_in_batches(data, optimal_batch)
        else:
            LogManager.log_info(self.name, "Using single-pass processing strategy")
            return self._process_all_frames(data)
    
    def _process_all_frames(self, data: VideoData) -> VideoData:
        """Process all frames at once."""
        # Convert VideoData frames to frame data format expected by scheduler
        frame_data_list = self._convert_to_frame_data_list(data)
        
        # Process frames using the scheduler's batch processing method
        try:
            # Create frame batch in the format expected by scheduler
            frame_batch = [(fd.frame_number, fd.rgb_matrix, fd.timestamp) for fd in frame_data_list]
            
            # Use the scheduler's batch processing method
            resized_frame_data = self.scheduler._resize_frame_batch(frame_batch)
              # Convert back to VideoData
            return self._convert_from_frame_data_list(resized_frame_data, data)
            
        except Exception as e:
            # Try fallback to CPU if GPU fails
            if self.use_gpu:
                self.logger.warning(f"GPU resize failed: {e}. Falling back to CPU...")
                LogManager.log_warning(self.name, f"GPU resize failed, falling back to CPU: {str(e)}")
                self._fallback_to_cpu()
                return self._process_all_frames(data)
            else:
                LogManager.log_error(self.name, f"Resize operation failed: {str(e)}", e)
                raise ProcessingError(
                    f"Resize operation failed: {str(e)}",
                    component=self.name,
                    details={'target_resolution': self.target_resolution}
                )
    
    def _fallback_to_cpu(self):
        """Fallback to CPU processing when GPU fails."""
        self.use_gpu = False
        # Recreate scheduler with CPU settings
        resize_func = self._get_resize_function()
        self.scheduler = VideoResizeTaskScheduler(
            resize_function=resize_func,
            target_width=self.target_resolution[0],
            target_height=self.target_resolution[1],
            max_workers=self.workers,
            use_cuda=False,
            device_id=self.gpu_device,
            execution_mode=self.execution_mode,            batch_size=self.batch_size,
            kernel_size=self.lanczos_kernel
        )
    
    def _process_in_batches(self, data: VideoData, batch_size: int) -> VideoData:
        """Process frames in batches to manage memory usage."""
        frame_count = data.frame_count
        all_resized_frames = []
        
        progress = ProgressReporter(frame_count, "Resizing frames")
        
        for start_idx in range(0, frame_count, batch_size):
            end_idx = min(start_idx + batch_size, frame_count)
            
            # Extract batch
            batch_frames = data.frames[start_idx:end_idx]
            batch_data = VideoData(
                frames=batch_frames,
                frame_rate=data.frame_rate,
                resolution=data.resolution,
                color_mode=data.color_mode,
                metadata=data.metadata.copy()
            )
            
            # Process batch
            resized_batch = self._process_all_frames(batch_data)
            all_resized_frames.append(resized_batch.frames)
            
            progress.update(end_idx - start_idx)
        
        progress.finish()
        
        # Concatenate all resized frames
        all_frames = np.concatenate(all_resized_frames, axis=0)
        
        # Create result VideoData
        result = VideoData(
            frames=all_frames,
            frame_rate=data.frame_rate,
            resolution=self.target_resolution,
            color_mode=data.color_mode,
            metadata=data.metadata.copy()
        )
        
        # Add processing step
        result.add_processing_step(self.name, {
            'target_resolution': self.target_resolution,
            'lanczos_kernel': self.lanczos_kernel,
            'use_gpu': self.use_gpu,
            'batch_processing': True,
            'batch_size': batch_size
        })
        
        return result
    
    def _convert_to_frame_data_list(self, data: VideoData) -> list:
        """Convert VideoData to list of FrameData objects."""
        frame_data_list = []
        
        for i, frame in enumerate(data.frames):
            # Ensure frame is in the right format
            if data.color_mode == 'RGB' and frame.ndim == 3 and frame.shape[2] == 3:
                rgb_matrix = frame
            elif data.color_mode == 'GRAY' and frame.ndim == 2:
                # Convert grayscale to RGB for processing
                rgb_matrix = np.stack([frame, frame, frame], axis=2)
            elif data.color_mode == 'GRAY' and frame.ndim == 3 and frame.shape[2] == 1:
                # Convert single-channel to RGB
                frame_2d = frame.squeeze(axis=2)
                rgb_matrix = np.stack([frame_2d, frame_2d, frame_2d], axis=2)
            else:
                raise ProcessingError(
                    f"Unsupported frame format: shape={frame.shape}, color_mode={data.color_mode}",
                    component=self.name
                )
            
            frame_data = FrameData(
                frame_number=i,
                rgb_matrix=rgb_matrix,
                timestamp=i / data.frame_rate,
                original_shape=frame.shape,
                target_shape=(*self.target_resolution[::-1], rgb_matrix.shape[2])  # (height, width, channels)
            )
            frame_data_list.append(frame_data)
        
        return frame_data_list
    
    def _convert_from_frame_data_list(self, frame_data_list: list, original_data: VideoData) -> VideoData:
        """Convert list of FrameData objects back to VideoData."""
        frames = []
        
        for frame_data in frame_data_list:
            if original_data.color_mode == 'RGB':
                frames.append(frame_data.rgb_matrix)
            elif original_data.color_mode == 'GRAY':
                # Convert RGB back to grayscale (take the red channel since all channels are the same)
                if frame_data.rgb_matrix.ndim == 3 and frame_data.rgb_matrix.shape[2] == 3:
                    gray_frame = frame_data.rgb_matrix[:, :, 0]
                else:
                    gray_frame = frame_data.rgb_matrix
                frames.append(gray_frame)
        
        frames_array = np.array(frames)
        
        # Create result VideoData
        result = VideoData(
            frames=frames_array,
            frame_rate=original_data.frame_rate,
            resolution=self.target_resolution,
            color_mode=original_data.color_mode,
            metadata=original_data.metadata.copy()
        )
        
        # Add processing step
        result.add_processing_step(self.name, {
            'target_resolution': self.target_resolution,
            'lanczos_kernel': self.lanczos_kernel,
            'use_gpu': self.use_gpu,
            'workers': self.workers,
            'execution_mode': self.execution_mode,
            'batch_size': self.batch_size
        })
        
        return result


# Convenience function for quick video resizing
def resize_video(data: VideoData,
                target_resolution: Tuple[int, int],
                lanczos_kernel: int = 4,
                use_gpu: bool = False,
                **kwargs) -> VideoData:
    """
    Convenience function to resize video data.
    
    Args:
        data: Input VideoData
        target_resolution: Target (width, height)
        lanczos_kernel: Lanczos kernel size
        use_gpu: Whether to use GPU acceleration
        **kwargs: Additional arguments for VideoResizer
        
    Returns:
        VideoData with resized frames
    """
    resizer = VideoResizer(target_resolution, lanczos_kernel, use_gpu, **kwargs)
    return resizer.process(data)
