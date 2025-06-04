"""
Component for decoding delta-encoded frames back into their original form.
Supports both CPU and GPU implementations with optional RLE decompression.
"""

import numpy as np
from typing import Dict, Optional, Any, Tuple
import time
from collections import deque

from . import ChainComponent, VideoData, ProcessingError, LogManager, ProgressReporter, MemoryManager

# Import CPU and GPU implementations
try:
    from ..cpu.delta_decode import compute_frame_decoding_cpu, decompress_deltas_rle_cpu
    CPU_AVAILABLE = True
except ImportError:
    CPU_AVAILABLE = False
    compute_frame_decoding_cpu = None
    decompress_deltas_rle_cpu = None

try:
    import cupy as cp
    from ..cuda.delta_decode import compute_frame_decoding_gpu, decompress_deltas_rle_gpu
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    compute_frame_decoding_gpu = None
    decompress_deltas_rle_gpu = None


class TemporalDeltaDecoder(ChainComponent):
    """Component for decoding delta-encoded frames back to their original form."""
    
    def __init__(self,
                 use_rle: bool = False,
                 use_gpu: bool = False,
                 gpu_device: int = 0,
                 workers: int = 4,
                 batch_size: int = 50):
        super().__init__("TemporalDeltaDecoder")
        
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
        
        self.use_rle = use_rle
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        self.workers = workers
        self.batch_size = batch_size
        
        # Performance tracking
        self.processing_times = deque(maxlen=10)
    
    def _validate_input(self, data: VideoData):
        """Validate input data for delta decoding."""
        super()._validate_input(data)
        
        if data.color_mode != 'BW':
            raise ProcessingError(
                f"Unsupported color mode: {data.color_mode}. Expected 1-bit BW",
                component=self.name
            )
        
        if data.frames.dtype != np.uint8:
            raise ProcessingError(
                f"Unsupported data type: {data.frames.dtype}. Expected uint8",
                component=self.name
            )
        
        # Verify frames are actually 1-bit (values 0 or 1)
        if not np.all(np.isin(data.frames, [0, 1])):
            raise ProcessingError(
                "Invalid pixel values: only 0 and 1 allowed for 1-bit frames",
                component=self.name
            )
        
        if data.frames.size == 0:
            raise ProcessingError(
                "No frames to process",
                component=self.name
            )
    
    def process(self, data: VideoData) -> VideoData:
        """
        Decode delta-encoded frames back to their original form.
        
        Args:
            data: Input VideoData with delta-encoded frames
            
        Returns:
            VideoData with decoded frames
        """
        self._validate_input(data)
        start_time = time.time()
        
        # Log operation start
        LogManager.log_info(
            self.name,
            f"Starting delta decoding: {data.frame_count} frames, "
            f"RLE={self.use_rle}, GPU={self.use_gpu}"
        )
        
        try:
            # Check memory usage and determine processing strategy
            memory_info = MemoryManager.estimate_memory_usage(data)
            LogManager.log_info(
                self.name,
                f"Memory analysis: {memory_info['total_mb']:.1f} MB for {data.frame_count} frames"
            )
            
            if MemoryManager.should_use_batch_processing(data):
                # Use memory manager to calculate optimal batch size
                final_batch = MemoryManager.calculate_optimal_batch_size(
                    data,
                    user_preference=self.batch_size
                )
                
                LogManager.log_info(
                    self.name,
                    f"Using batch processing strategy: batch_size={final_batch}"
                )
                
                processed_data = self._process_in_batches(data, final_batch)
            else:
                LogManager.log_info(self.name, "Using single-pass processing strategy")
                processed_data = self._process_all_frames(data)
            
            # Calculate processing statistics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Log completion
            LogManager.log_info(
                self.name,
                f"Delta decoding complete: {processing_time:.2f}s processing time"
            )
            
            return processed_data
            
        except Exception as e:
            LogManager.log_error(self.name, f"Delta decoding failed: {str(e)}", e)
            raise ProcessingError(f"Delta decoding failed: {str(e)}", component=self.name)
    
    def _process_all_frames(self, data: VideoData) -> VideoData:
        """Process all frames at once."""
        try:
            # Process using optimized implementation
            if self.use_gpu and CUDA_AVAILABLE:
                with cp.cuda.Device(self.gpu_device):
                    # Transfer data to GPU
                    frames_gpu = cp.asarray(data.frames)
                    
                    # Optional RLE decompression
                    if self.use_rle:
                        frames_gpu = decompress_deltas_rle_gpu(frames_gpu)
                    
                    # Compute decoded frames
                    result_frames = compute_frame_decoding_gpu(frames_gpu)
                    
                    # Transfer back to CPU
                    result_frames = cp.asnumpy(result_frames)
            else:
                # Optional RLE decompression
                frames = data.frames
                if self.use_rle:
                    frames = decompress_deltas_rle_cpu(frames, workers=self.workers)
                
                # Decode frames
                result_frames = compute_frame_decoding_cpu(frames, workers=self.workers)
            
            # Create output VideoData
            result = VideoData(
                frames=result_frames,
                frame_rate=data.frame_rate,
                resolution=data.resolution,
                color_mode='BW',
                metadata=data.metadata.copy()
            )
            
            # Add processing step to metadata
            result.add_processing_step(self.name, {
                'use_rle': self.use_rle,
                'use_gpu': self.use_gpu,
                'processing_time_seconds': np.mean(list(self.processing_times))
            })
            
            return result
            
        except Exception as e:
            LogManager.log_error(self.name, f"Frame processing failed: {str(e)}", e)
            raise ProcessingError(
                f"Frame processing failed: {str(e)}",
                component=self.name
            )
    
    def _process_in_batches(self, data: VideoData, batch_size: int) -> VideoData:
        """Process frames in batches to manage memory usage."""
        frame_count = data.frame_count
        all_processed_frames = []
        
        progress = ProgressReporter(frame_count, "Delta decoding frames")
        
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
            processed_batch = self._process_all_frames(batch_data)
            all_processed_frames.append(processed_batch.frames)
            
            progress.update(end_idx - start_idx)
        
        progress.finish()
        
        # Concatenate all processed frames
        all_frames = np.concatenate(all_processed_frames, axis=0)
        
        # Create result VideoData
        result = VideoData(
            frames=all_frames,
            frame_rate=data.frame_rate,
            resolution=data.resolution,
            color_mode='BW',
            metadata=data.metadata.copy()
        )
        
        # Add processing step to metadata
        result.add_processing_step(self.name, {
            'use_rle': self.use_rle,
            'use_gpu': self.use_gpu,
            'processing_time_seconds': np.mean(list(self.processing_times)),
            'batch_processing': True,
            'batch_size': batch_size
        })
        
        return result
