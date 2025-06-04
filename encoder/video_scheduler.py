"""
Video processing scheduler using PyAV for video handling and parallel processing.
Supports both CPU and CUDA Lanczos resizing with configurable kernel sizes.
"""

import av
import numpy as np
from dataclasses import dataclass
from typing import List, Callable, Tuple, Optional, Union, Any
from pathlib import Path
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue
import warnings


@dataclass
class FrameData:
    """Container for frame data with RGB matrix and metadata."""
    frame_number: int
    rgb_matrix: np.ndarray  # Shape: (height, width, 3) for color or (height, width) for grayscale
    timestamp: float
    original_shape: Tuple[int, ...]  # (height, width) or (height, width, channels)
    target_shape: Tuple[int, ...]    # (height, width) or (height, width, channels)


@dataclass
class VideoMetadata:
    """Container for video metadata."""
    width: int
    height: int
    fps: float
    duration: float
    frame_count: int
    codec: str
    pixel_format: str


class VideoValidator:
    """Validates video files and extracts metadata using PyAV."""
    
    SUPPORTED_FORMATS = {
        'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv', 'm4v'
    }
    
    SUPPORTED_CODECS = {
        'h264', 'h265', 'hevc', 'vp8', 'vp9', 'av1', 'mpeg4', 'xvid'
    }
    
    @staticmethod
    def validate_video_file(video_path: Union[str, Path]) -> VideoMetadata:
        """
        Validate video file and extract metadata.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            VideoMetadata object with file information
            
        Raises:
            ValueError: If file is invalid or unsupported
            FileNotFoundError: If file doesn't exist
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if not video_path.is_file():
            raise ValueError(f"Path is not a file: {video_path}")
        
        # Check file extension
        extension = video_path.suffix.lower().lstrip('.')
        if extension not in VideoValidator.SUPPORTED_FORMATS:
            warnings.warn(f"File extension '{extension}' might not be supported. Supported: {VideoValidator.SUPPORTED_FORMATS}")
        
        try:
            # Open video with PyAV
            container = av.open(str(video_path))
            
            # Get video stream
            video_stream = container.streams.video[0]
            if video_stream is None:
                raise ValueError("No video stream found in file")
              # Extract metadata
            duration_seconds = 0.0
            if video_stream.duration and video_stream.time_base:
                duration_seconds = float(video_stream.duration * video_stream.time_base)
            
            metadata = VideoMetadata(
                width=video_stream.width,
                height=video_stream.height,
                fps=float(video_stream.average_rate) if video_stream.average_rate else 30.0,
                duration=duration_seconds,
                frame_count=video_stream.frames if video_stream.frames else 0,
                codec=video_stream.codec.name.lower(),
                pixel_format=video_stream.pix_fmt or 'unknown'
            )
            
            container.close()
            
            # Validate codec
            if metadata.codec not in VideoValidator.SUPPORTED_CODECS:
                warnings.warn(f"Codec '{metadata.codec}' might not be well supported. Recommended: {VideoValidator.SUPPORTED_CODECS}")
            
            # Basic sanity checks
            if metadata.width <= 0 or metadata.height <= 0:
                raise ValueError(f"Invalid video dimensions: {metadata.width}x{metadata.height}")
            
            if metadata.fps <= 0:
                warnings.warn(f"Invalid or missing FPS: {metadata.fps}, defaulting to 30.0")
                metadata.fps = 30.0
            return metadata

        except Exception as e:
            if "av" in str(type(e)).lower():
                raise ValueError(f"Failed to open video file: {e}")
            else:
                raise ValueError(f"Unexpected error reading video metadata: {e}")


class VideoFrameExtractor:
    """Extracts frames from video using PyAV."""
    
    def __init__(self, video_path: Union[str, Path], convert_to_rgb: bool = True):
        """
        Initialize frame extractor.
        
        Args:
            video_path: Path to video file
            convert_to_rgb: Whether to convert frames to RGB format
        """
        self.video_path = Path(video_path)
        self.convert_to_rgb = convert_to_rgb
        self._container = None
        self._video_stream = None
        
    def __enter__(self):
        """Context manager entry."""
        self._container = av.open(str(self.video_path))
        self._video_stream = self._container.streams.video[0]
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._container:
            self._container.close()
            
    def extract_frames(self, max_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        Extract all frames from video.
        
        Args:
            max_frames: Maximum number of frames to extract (None for all)
            
        Returns:
            List of numpy arrays representing frames
        """
        if not self._container:
            raise RuntimeError("Extractor not initialized. Use as context manager.")
        
        frames = []
        frame_count = 0
        
        for frame in self._container.decode(self._video_stream):
            if max_frames and frame_count >= max_frames:
                break
                
            # Convert frame to numpy array
            if self.convert_to_rgb:
                # Convert to RGB format
                frame_rgb = frame.to_rgb()
                frame_array = frame_rgb.to_ndarray()
            else:
                # Keep original format
                frame_array = frame.to_ndarray()
            
            frames.append(frame_array)
            frame_count += 1
            
        return frames
    
    def extract_frames_generator(self, max_frames: Optional[int] = None):
        """
        Generator that yields frames one by one.
        
        Args:
            max_frames: Maximum number of frames to extract (None for all)
            
        Yields:
            Tuple of (frame_number, numpy_array, timestamp)
        """
        if not self._container:
            raise RuntimeError("Extractor not initialized. Use as context manager.")
        
        frame_count = 0
        for frame in self._container.decode(self._video_stream):
            if max_frames and frame_count >= max_frames:
                break
                
            # Skip non-video frames
            if not hasattr(frame, 'to_rgb') or not hasattr(frame, 'to_ndarray'):
                continue
                
            # Convert frame to numpy array
            if self.convert_to_rgb:
                frame_rgb = frame.to_rgb()
                frame_array = frame_rgb.to_ndarray()
            else:
                frame_array = frame.to_ndarray()
            
            # Calculate timestamp
            timestamp = 0.0
            if frame.pts and self._video_stream and self._video_stream.time_base:
                timestamp = float(frame.pts * self._video_stream.time_base)
            
            yield frame_count, frame_array, timestamp
            frame_count += 1


class VideoResizeTaskScheduler:
    """
    Parallel video resizing scheduler supporting CPU and CUDA Lanczos resizing.
    Uses PyAV for video handling and configurable parallel processing.
    """
    def __init__(
        self,
        resize_function: Callable,
        target_width: int,
        target_height: int,
        max_workers: int = 4,
        use_cuda: bool = False,
        device_id: int = 0,
        execution_mode: str = 'threading',
        batch_size: int = 50,
        kernel_size: int = 4
    ):
        """
        Initialize the video resize scheduler.
        
        Args:
            resize_function: Function to use for resizing (CPU or CUDA)
            target_width: Target width for resized frames
            target_height: Target height for resized frames
            max_workers: Maximum number of worker threads/processes
            use_cuda: Whether to use CUDA acceleration
            device_id: CUDA device ID to use
            execution_mode: 'threading' or 'multiprocessing'
            batch_size: Number of frames to process in each batch
            kernel_size: Lanczos kernel size (2, 3, 4, etc.)
        """
        self.resize_function = resize_function
        self.target_width = target_width
        self.target_height = target_height
        self.max_workers = max_workers
        self.use_cuda = use_cuda
        self.device_id = device_id
        self.execution_mode = execution_mode
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        
        # Validate parameters
        if execution_mode not in ['threading', 'multiprocessing']:
            raise ValueError("execution_mode must be 'threading' or 'multiprocessing'")
        
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        
        if target_width <= 0 or target_height <= 0:
            raise ValueError("Target dimensions must be positive")
        
        if kernel_size < 2:
            raise ValueError("Kernel size must be at least 2")
    
    def _resize_frame_batch(self, frame_batch: List[Tuple[int, np.ndarray, float]]) -> List[FrameData]:
        """
        Resize a batch of frames.
        
        Args:
            frame_batch: List of (frame_number, frame_array, timestamp) tuples
            
        Returns:
            List of FrameData objects with resized frames
        """
        results = []
        
        for frame_number, frame_array, timestamp in frame_batch:
            original_shape = frame_array.shape
            
            # Determine if frame is color or grayscale
            if len(original_shape) == 3:
                channels = original_shape[2]
                target_shape = (self.target_height, self.target_width, channels)
            else:
                channels = 1
                target_shape = (self.target_height, self.target_width)
              # Resize frame
            try:
                # Check if resize function supports kernel_size parameter
                import inspect
                sig = inspect.signature(self.resize_function)
                if 'kernel_size' in sig.parameters:
                    # Use new configurable resize function
                    resized_frame = self.resize_function(
                        frame_array, 
                        self.target_width, 
                        self.target_height,
                        kernel_size=self.kernel_size
                    )
                else:
                    # Use legacy fixed Lanczos-4 function
                    resized_frame = self.resize_function(
                        frame_array, 
                        self.target_width, 
                        self.target_height
                    )
                  # Handle CuPy to NumPy conversion if needed
                if hasattr(resized_frame, 'get'):
                    # This is a CuPy array, convert to NumPy
                    resized_frame = resized_frame.get()
                
                # Create FrameData object
                frame_data = FrameData(
                    frame_number=frame_number,
                    rgb_matrix=resized_frame,
                    timestamp=timestamp,
                    original_shape=original_shape,
                    target_shape=target_shape
                )
                
                results.append(frame_data)
                
            except Exception as e:
                raise RuntimeError(f"Failed to resize frame {frame_number}: {e}")
        
        return results
    
    def process_video_file(
        self, 
        video_path: Union[str, Path], 
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        max_frames: Optional[int] = None
    ) -> Tuple[List[FrameData], VideoMetadata]:
        """
        Process entire video file with resizing.
        
        Args:
            video_path: Path to input video file
            progress_callback: Optional callback for progress updates (current, total, elapsed_time)
            max_frames: Maximum number of frames to process (None for all)
            
        Returns:
            Tuple of (list_of_resized_frames, video_metadata)
        """
        # Validate video file
        metadata = VideoValidator.validate_video_file(video_path)
        
        # Determine actual frame count to process
        total_frames = max_frames if max_frames else metadata.frame_count
        if total_frames <= 0:
            # Fallback: estimate from duration and fps
            total_frames = max_frames if max_frames else int(metadata.duration * metadata.fps)
        
        print(f"📹 Processing {total_frames} frames from {video_path}")
        print(f"🔧 Using {self.execution_mode} with {self.max_workers} workers")
        print(f"📦 Batch size: {self.batch_size}")
        print(f"🎯 Target resolution: {self.target_width}x{self.target_height}")
        print(f"🔍 Lanczos kernel size: {self.kernel_size}")
        
        start_time = time.perf_counter()
        all_results = []
        processed_frames = 0
        
        # Extract frames in batches and process them
        with VideoFrameExtractor(video_path, convert_to_rgb=True) as extractor:
            frame_batch = []
            
            for frame_number, frame_array, timestamp in extractor.extract_frames_generator(max_frames):
                frame_batch.append((frame_number, frame_array, timestamp))
                
                # Process batch when full
                if len(frame_batch) >= self.batch_size:
                    batch_results = self._process_batch_parallel(frame_batch)
                    all_results.extend(batch_results)
                    processed_frames += len(frame_batch)
                    
                    # Progress callback
                    if progress_callback:
                        elapsed_time = time.perf_counter() - start_time
                        progress_callback(processed_frames, total_frames, elapsed_time)
                    
                    frame_batch = []
            
            # Process remaining frames
            if frame_batch:
                batch_results = self._process_batch_parallel(frame_batch)
                all_results.extend(batch_results)
                processed_frames += len(frame_batch)
                
                # Final progress callback
                if progress_callback:
                    elapsed_time = time.perf_counter() - start_time
                    progress_callback(processed_frames, total_frames, elapsed_time)
        
        # Sort results by frame number to maintain order
        all_results.sort(key=lambda x: x.frame_number)
        
        return all_results, metadata
    
    def _process_batch_parallel(self, frame_batch: List[Tuple[int, np.ndarray, float]]) -> List[FrameData]:
        """
        Process a batch of frames using parallel execution.
        
        Args:
            frame_batch: List of frame tuples to process
            
        Returns:
            List of processed FrameData objects
        """
        if self.execution_mode == 'threading':
            # Use ThreadPoolExecutor for threading
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Split batch into sub-batches for workers
                sub_batches = self._split_batch(frame_batch, self.max_workers)
                
                # Submit tasks
                futures = [
                    executor.submit(self._resize_frame_batch, sub_batch)
                    for sub_batch in sub_batches if sub_batch
                ]
                
                # Collect results
                results = []
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        results.extend(batch_results)
                    except Exception as e:
                        raise RuntimeError(f"Batch processing failed: {e}")
                
                return results
        
        else:  # multiprocessing
            # Use ProcessPoolExecutor for multiprocessing
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Split batch into sub-batches for workers
                sub_batches = self._split_batch(frame_batch, self.max_workers)
                
                # Submit tasks
                futures = [
                    executor.submit(self._resize_frame_batch, sub_batch)
                    for sub_batch in sub_batches if sub_batch
                ]
                
                # Collect results
                results = []
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        results.extend(batch_results)
                    except Exception as e:
                        raise RuntimeError(f"Batch processing failed: {e}")
                
                return results
    
    def _split_batch(self, batch: List, num_splits: int) -> List[List]:
        """Split a batch into roughly equal sub-batches."""
        if not batch:
            return []
        
        batch_size = len(batch)
        sub_batch_size = max(1, batch_size // num_splits)
        
        sub_batches = []
        for i in range(0, batch_size, sub_batch_size):
            sub_batch = batch[i:i + sub_batch_size]
            if sub_batch:
                sub_batches.append(sub_batch)
        
        return sub_batches


def export_frames_as_numpy_array(frames: List[FrameData]) -> np.ndarray:
    """
    Export processed frames as a single numpy array.
    
    Args:
        frames: List of FrameData objects
        
    Returns:
        Numpy array with shape (num_frames, height, width, channels)
    """
    if not frames:
        raise ValueError("No frames to export")
    
    # Determine array shape from first frame
    first_frame = frames[0]
    if len(first_frame.rgb_matrix.shape) == 3:
        height, width, channels = first_frame.rgb_matrix.shape
        array_shape = (len(frames), height, width, channels)
    else:
        height, width = first_frame.rgb_matrix.shape
        array_shape = (len(frames), height, width)
    
    # Create output array
    dtype = first_frame.rgb_matrix.dtype
    output_array = np.zeros(array_shape, dtype=dtype)
    
    # Copy frames
    for i, frame in enumerate(frames):
        output_array[i] = frame.rgb_matrix
    
    return output_array


def save_resized_frames_to_video(
    frames: List[FrameData], 
    output_path: Union[str, Path], 
    fps: float,
    codec: str = 'libx264',
    pixel_format: str = 'yuv420p'
) -> None:
    """
    Save resized frames as a video file using PyAV.
    
    Args:
        frames: List of FrameData objects
        output_path: Output video file path
        fps: Frame rate for output video
        codec: Video codec to use
        pixel_format: Pixel format for output video
    """
    if not frames:
        raise ValueError("No frames to save")
    
    output_path = Path(output_path)
    
    # Get frame dimensions from first frame
    first_frame = frames[0]
    if len(first_frame.rgb_matrix.shape) == 3:
        height, width, channels = first_frame.rgb_matrix.shape
    else:
        height, width = first_frame.rgb_matrix.shape
        channels = 1
      # Create output container
    container = av.open(str(output_path), mode='w')
    
    # Create video stream
    from fractions import Fraction
    fps_fraction = Fraction(fps).limit_denominator()
    stream = container.add_stream(codec, rate=fps_fraction)
    stream.width = width
    stream.height = height
    stream.pix_fmt = pixel_format
    
    try:
        # Write frames
        for frame_data in frames:
            # Create PyAV frame
            if channels == 3:
                # RGB frame
                frame = av.VideoFrame.from_ndarray(frame_data.rgb_matrix, format='rgb24')
            else:
                # Grayscale frame - convert to RGB for compatibility
                rgb_frame = np.stack([frame_data.rgb_matrix] * 3, axis=-1)
                frame = av.VideoFrame.from_ndarray(rgb_frame, format='rgb24')
            
            # Encode and write
            for packet in stream.encode(frame):
                container.mux(packet)
        
        # Flush encoder
        for packet in stream.encode():
            container.mux(packet)
            
    finally:
        container.close()


__all__ = [
    'VideoValidator', 
    'VideoFrameExtractor', 
    'VideoResizeTaskScheduler', 
    'FrameData', 
    'VideoMetadata',
    'export_frames_as_numpy_array',
    'save_resized_frames_to_video'
]
