"""
1-bit video conversion component with optional dithering support.

This module provides RGB/Grayscale to 1-bit conversion with multiple dithering
algorithms for improved visual quality. Supports both CPU and GPU acceleration.
"""

import numpy as np
from typing import Optional, Literal, Dict, cast, TypeVar, Any, Union, overload
import time
from collections import deque

from .basex import ChainComponent, VideoData, ProcessingError, LogManager

# Import CPU and GPU implementations
try:
    from ..cpu.convert1bit import convert_1bit_cpu
    CPU_AVAILABLE = True
except ImportError:
    CPU_AVAILABLE = False
    convert_1bit_cpu = None

try:
    import cupy as cp
    from ..cuda.convert1bit import convert_1bit_gpu
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    convert_1bit_gpu = None

# Type aliases for better readability
DitherMethod = Literal['none', 'floyd-steinberg', 'ordered']
DitherPattern = Literal['bayer2x2', 'bayer4x4', 'bayer8x8']

class Video1BitConverter(ChainComponent):
    """Component for converting RGB/Grayscale video to 1-bit with optional dithering."""
    
    def __init__(self,
                 threshold: int = 128,
                 dither_method: Union[DitherMethod, None] = 'none',
                 dither_pattern: DitherPattern = 'bayer4x4',
                 use_gpu: bool = False,
                 gpu_device: int = 0):
        """
        Initialize the 1-bit converter.
        
        Args:
            threshold: Threshold value for binary conversion (0-255)
            dither_method: Dithering method to use ('none', 'floyd-steinberg', 'ordered')
            dither_pattern: Pattern for ordered dithering
            use_gpu: Whether to use GPU acceleration
            gpu_device: GPU device ID to use
        """
        super().__init__("Video1BitConverter")
        
        # Validation
        if not 0 <= threshold <= 255:
            raise ValueError("Threshold must be between 0 and 255")
            
        if dither_method is not None and dither_method not in ('none', 'floyd-steinberg', 'ordered'):
            raise ValueError("dither_method must be one of: none, floyd-steinberg, ordered")
            
        if dither_pattern not in ('bayer2x2', 'bayer4x4', 'bayer8x8'):
            raise ValueError("dither_pattern must be one of: bayer2x2, bayer4x4, bayer8x8")
        
        # Check implementation availability
        if use_gpu and not CUDA_AVAILABLE:
            LogManager.log_warning(self.name, "GPU requested but CUDA not available, falling back to CPU")
            use_gpu = False
        elif not use_gpu and not CPU_AVAILABLE:
            if CUDA_AVAILABLE:
                LogManager.log_warning(self.name, "CPU implementation not available, using GPU instead")
                use_gpu = True
            else:
                raise RuntimeError("Neither CPU nor GPU implementations are available")
        self.threshold = threshold
        self.dither_method: Union[DitherMethod, None] = dither_method
        self.dither_pattern: DitherPattern = dither_pattern
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        
        # Performance tracking
        self.processing_times = deque(maxlen=10)
        
        # Log warning if dither pattern is provided but won't be used
        if dither_method != 'ordered' and dither_pattern != 'bayer4x4':
            LogManager.log_warning(
                self.name,
                f"Dither pattern '{dither_pattern}' will be ignored as dither method is not 'ordered'"
            )
        
        LogManager.log_info(
            self.name,
            f"Initialized: threshold={threshold}, dither={dither_method}, GPU={use_gpu}"
        )

    def _validate_input(self, data: VideoData):
        """Validate input data for 1-bit conversion."""
        super()._validate_input(data)
        
        if data.color_mode not in ['RGB', 'GRAY']:
            raise ProcessingError(
                f"Unsupported color mode: {data.color_mode}. Expected 'RGB' or 'GRAY'",
                component=self.name
            )
        
        if data.frames.size == 0:
            raise ProcessingError(
                "No frames to process",
                component=self.name
            )

    def process(self, data: VideoData) -> VideoData:
        """
        Convert video frames to 1-bit with optional dithering.
        
        Args:
            data: Input VideoData
            
        Returns:
            VideoData with 1-bit frames and updated metadata
        """
        self._validate_input(data)
        start_time = time.time()
        
        LogManager.log_info(
            self.name,
            f"Starting 1-bit conversion: {data.frame_count} frames, "
            f"method={self.dither_method}, threshold={self.threshold}"
        )
        
        try:
            # Convert frames using optimized implementation
            if self.use_gpu and CUDA_AVAILABLE:
                binary_frames = convert_1bit_gpu(
                    data.frames,
                    threshold=self.threshold,
                    dither_method=self.dither_method,
                    dither_pattern=self.dither_pattern,
                    device_id=self.gpu_device
                )
            else:
                binary_frames = convert_1bit_cpu(
                    data.frames,
                    threshold=self.threshold,
                    dither_method=self.dither_method,
                    dither_pattern=self.dither_pattern
                )
            
            # Calculate processing statistics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Calculate conversion statistics
            original_pixels = np.prod(binary_frames.shape)
            white_pixels = np.sum(binary_frames == 1)
            black_pixels = np.sum(binary_frames == 0)
            white_percentage = (white_pixels / original_pixels) * 100
            
            LogManager.log_info(
                self.name,
                f"Conversion complete: {white_percentage:.1f}% white pixels, "
                f"{processing_time:.2f}s processing time"
            )
            
            # Create output VideoData
            output_data = VideoData(
                frames=binary_frames,
                frame_rate=data.frame_rate,
                resolution=data.resolution,
                color_mode='BW',
                metadata=data.metadata.copy()
            )
            
            # Add processing step to metadata
            output_data.add_processing_step(self.name, {
                'threshold': self.threshold,
                'dither_method': self.dither_method,
                'dither_pattern': self.dither_pattern if self.dither_method == 'ordered' else None,
                'use_gpu': self.use_gpu,
                'processing_time_seconds': processing_time,
                'white_pixel_percentage': white_percentage,
                'conversion_stats': {
                    'total_pixels': int(original_pixels),
                    'white_pixels': int(white_pixels),
                    'black_pixels': int(black_pixels)
                }
            })
            
            return output_data
            
        except Exception as e:
            LogManager.log_error(self.name, f"1-bit conversion failed: {str(e)}", e)
            raise ProcessingError(f"1-bit conversion failed: {str(e)}", component=self.name)
