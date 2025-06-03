"""
Video file opening and frame extraction component.

This component handles video file validation and frame extraction using PyAV,
converting video files into standardized VideoData structures for processing.
"""

import av
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union
import logging

from . import ChainComponent, VideoData, ProcessingError, ProgressReporter, LogManager


class VideoOpener(ChainComponent):
    """Component for opening video files and extracting frames."""
    
    SUPPORTED_FORMATS = {
        'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv', 'm4v'
    }
    
    def __init__(self, 
                 file_path: Union[str, Path],
                 color_mode: str = 'RGB',
                 max_frames: Optional[int] = None):
        """
        Initialize VideoOpener component.
        
        Args:
            file_path: Path to the video file
            color_mode: Output color mode ('RGB' or 'GRAY')
            max_frames: Maximum number of frames to extract (None for all)
        """
        super().__init__("VideoOpener")
        
        self.file_path = Path(file_path)
        self.color_mode = color_mode.upper()
        self.max_frames = max_frames
        
        if self.color_mode not in ['RGB', 'GRAY']:
            raise ValueError(f"Unsupported color mode: {color_mode}. Use 'RGB' or 'GRAY'")
        
        # Validate file
        self._validate_file()
    
    def _validate_file(self):
        """Validate the video file exists and is supported."""
        if not self.file_path.exists():
            raise ProcessingError(
                f"Video file does not exist: {self.file_path}",
                component=self.name
            )
        
        if not self.file_path.is_file():
            raise ProcessingError(
                f"Path is not a file: {self.file_path}",
                component=self.name
            )
        
        # Check file extension
        file_extension = self.file_path.suffix.lower().lstrip('.')
        if file_extension not in self.SUPPORTED_FORMATS:
            self.logger.warning(
                f"File extension '.{file_extension}' not in supported formats: {self.SUPPORTED_FORMATS}. "
                "Attempting to open anyway..."
            )
    
    def _validate_input(self, data: VideoData):
        """Override input validation - VideoOpener doesn't take VideoData as input."""
        # VideoOpener is typically the first component, so it may not receive VideoData
        pass
    
    def process(self, data: Optional[VideoData] = None) -> VideoData:
        """
        Extract frames from the video file.
        
        Args:
            data: Not used by VideoOpener (it's typically the first component)
            
        Returns:
            VideoData containing extracted frames
        """
        try:
            # Open video file
            container = av.open(str(self.file_path))
            video_stream = container.streams.video[0]
              # Extract metadata
            width = video_stream.width
            height = video_stream.height
            frame_rate = float(video_stream.average_rate) if video_stream.average_rate else 30.0
            
            if video_stream.duration and video_stream.time_base:
                duration = float(video_stream.duration * video_stream.time_base)
            else:
                duration = 0
            
            # Estimate frame count
            if video_stream.frames:
                estimated_frames = video_stream.frames
            else:
                estimated_frames = int(duration * frame_rate) if duration > 0 else 1000
            
            # Apply max_frames limit
            if self.max_frames:
                estimated_frames = min(estimated_frames, self.max_frames)
            
            self.logger.info(f"Video: {width}x{height}, {frame_rate:.2f} fps, ~{estimated_frames} frames")
            LogManager.log_info(self.name, f"Video metadata: {width}x{height}, {frame_rate:.2f} fps, ~{estimated_frames} frames")
            
            # Extract frames
            frames = self._extract_frames(container, video_stream, estimated_frames)
            container.close()
            
            # Create VideoData
            video_data = VideoData(
                frames=frames,
                frame_rate=frame_rate,
                resolution=(width, height),
                color_mode=self.color_mode,
                metadata={
                    'source_file': str(self.file_path),
                    'original_duration': duration,
                    'codec': video_stream.codec.name if video_stream.codec else 'unknown',
                    'pixel_format': str(video_stream.pix_fmt) if video_stream.pix_fmt else 'unknown'
                }
            )
              # Add processing step
            video_data.add_processing_step(self.name, {
                'file_path': str(self.file_path),
                'color_mode': self.color_mode,
                'max_frames': self.max_frames
            })
            
            LogManager.log_info(self.name, f"Successfully extracted {video_data.frame_count} frames from {self.file_path}")
            return video_data
            
        except Exception as av_error:
            # Handle both av.AVError and other exceptions
            error_msg = f"Error while opening video: {str(av_error)}"
            if "av." in str(type(av_error)):
                error_msg = f"FFmpeg/AVError while opening video: {str(av_error)}"
            
            # Log error with full traceback using LogManager
            LogManager.log_error(self.name, error_msg, av_error)
            
            raise ProcessingError(
                error_msg,
                component=self.name,
                details={'file_path': str(self.file_path), 'av_error': str(av_error)}
            )
    
    def _extract_frames(self, container, video_stream, estimated_frames: int) -> np.ndarray:
        """Extract frames from the video stream."""
        frames_list = []
        progress = ProgressReporter(estimated_frames, f"Extracting frames")
        
        frame_count = 0
        for frame in container.decode(video_stream):
            if self.max_frames and frame_count >= self.max_frames:
                break
            
            # Convert frame to numpy array
            if self.color_mode == 'RGB':
                # Convert to RGB format
                frame_rgb = frame.to_rgb()
                frame_array = frame_rgb.to_ndarray()
            else:  # GRAY
                # Convert to grayscale
                frame_gray = frame.to_gray()
                frame_array = frame_gray.to_ndarray()
                # Ensure 2D array for grayscale
                if frame_array.ndim == 3 and frame_array.shape[2] == 1:
                    frame_array = frame_array.squeeze(axis=2)
            
            frames_list.append(frame_array)
            frame_count += 1
            
            # Update progress
            if frame_count % max(1, estimated_frames // 20) == 0:
                progress.update(max(1, estimated_frames // 20))
        
        progress.finish()
        
        if not frames_list:
            raise ProcessingError(
                "No frames extracted from video",
                component=self.name,
                details={'file_path': str(self.file_path)}
            )
        
        # Convert to numpy array
        frames_array = np.array(frames_list)
        self.logger.info(f"Extracted {len(frames_list)} frames, shape: {frames_array.shape}")
        LogManager.log_info(self.name, f"Frame extraction completed: {len(frames_list)} frames, shape: {frames_array.shape}")
        
        return frames_array
    
    def open(self) -> VideoData:
        """
        Convenience method to extract frames without chaining.
        
        Returns:
            VideoData containing extracted frames
        """
        return self.process()


# Convenience function for quick video opening
def open_video(file_path: Union[str, Path], 
               color_mode: str = 'RGB',
               max_frames: Optional[int] = None) -> VideoData:
    """
    Convenience function to open a video file and extract frames.
    
    Args:
        file_path: Path to the video file
        color_mode: Output color mode ('RGB' or 'GRAY')
        max_frames: Maximum number of frames to extract (None for all)
        
    Returns:
        VideoData containing extracted frames
    """
    opener = VideoOpener(file_path, color_mode, max_frames)
    return opener.open()