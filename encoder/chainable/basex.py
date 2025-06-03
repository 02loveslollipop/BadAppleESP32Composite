"""
Base architecture for chainable video processing components.

This module provides the foundational classes and data structures for the
chainable video processing pipeline with enhanced logging and traceback support.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
from abc import ABC, abstractmethod
import logging
import traceback
import sys
from pathlib import Path
from datetime import datetime


@dataclass
class VideoData:
    """Standardized data structure for video processing pipeline."""
    frames: np.ndarray  # Video frame data: (frame_count, height, width, channels) or (frame_count, height, width)
    frame_rate: float
    resolution: Tuple[int, int]  # (width, height)
    color_mode: str  # 'RGB', 'GRAY', 'BW'
    metadata: Dict[str, Any] = field(default_factory=dict)  # Processing history and parameters
    
    def __post_init__(self):
        """Validate VideoData after initialization."""
        if self.frames.ndim not in [3, 4]:
            raise ValueError(f"Frames must be 3D or 4D array, got {self.frames.ndim}D")
        
        # Validate color mode against frame dimensions
        if self.color_mode == 'RGB' and (self.frames.ndim != 4 or self.frames.shape[-1] != 3):
            raise ValueError("RGB color mode requires 4D array with 3 channels")
        elif self.color_mode in ['GRAY', 'BW'] and self.frames.ndim not in [3, 4]:
            raise ValueError(f"{self.color_mode} color mode requires 3D or 4D array")
        
        # Validate resolution matches frame dimensions
        if self.frames.ndim == 4:
            frame_height, frame_width = self.frames.shape[1:3]
        else:
            frame_height, frame_width = self.frames.shape[1:3]
        if (frame_width, frame_height) != self.resolution:
            raise ValueError(f"Resolution {self.resolution} doesn't match frame dimensions ({frame_width}, {frame_height})")
    
    @property
    def frame_count(self) -> int:
        """Get the number of frames."""
        return self.frames.shape[0]
    
    @property
    def width(self) -> int:
        """Get frame width."""
        return self.resolution[0]
    
    @property
    def height(self) -> int:
        """Get frame height."""
        return self.resolution[1]
    
    def add_processing_step(self, component_name: str, parameters: Dict[str, Any]):
        """Add a processing step to the metadata history."""
        if 'processing_history' not in self.metadata:
            self.metadata['processing_history'] = []
        
        self.metadata['processing_history'].append({
            'component': component_name,
            'parameters': parameters.copy(),
            'timestamp': np.datetime64('now').astype(str)
        })


class LogManager:
    """Manages detailed logging for chainable components with full traceback support."""
    
    _log_file_path: Optional[Path] = None
    _file_handler: Optional[logging.FileHandler] = None
    _initialized = False
    
    @classmethod
    def initialize(cls, log_dir: str = "logs"):
        """Initialize the log manager with a clean log file for this processing run."""
        if cls._initialized:
            cls.cleanup()
        
        # Create logs directory
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cls._log_file_path = log_path / f"chainable_processing_{timestamp}.log"
        
        # Create file handler
        cls._file_handler = logging.FileHandler(cls._log_file_path, mode='w', encoding='utf-8')
        cls._file_handler.setLevel(logging.DEBUG)
        
        # Detailed formatter for file logs
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        cls._file_handler.setFormatter(file_formatter)
        
        cls._initialized = True
        
        # Log initialization
        root_logger = logging.getLogger('chainable')
        root_logger.addHandler(cls._file_handler)
        root_logger.setLevel(logging.DEBUG)
        
        cls.log_info("LogManager", f"Initialized logging to: {cls._log_file_path}")
    
    @classmethod
    def cleanup(cls):
        """Clean up logging resources."""
        if cls._file_handler:
            # Remove handler from all loggers
            for logger_name in logging.Logger.manager.loggerDict:
                if logger_name.startswith('chainable'):
                    logger = logging.getLogger(logger_name)
                    if cls._file_handler in logger.handlers:
                        logger.removeHandler(cls._file_handler)
            
            cls._file_handler.close()
            cls._file_handler = None
        
        cls._initialized = False
    
    @classmethod
    def log_info(cls, component: str, message: str):
        """Log an info message."""
        if cls._initialized:
            logger = logging.getLogger(f'chainable.{component}')
            logger.info(message)
    
    @classmethod
    def log_error(cls, component: str, message: str, exception: Optional[Exception] = None):
        """Log an error message with full traceback."""
        if cls._initialized:
            logger = logging.getLogger(f'chainable.{component}')
            logger.error(message)
            
            if exception:
                # Log full traceback
                tb_str = traceback.format_exc()
                logger.error(f"Full traceback:\n{tb_str}")
    
    @classmethod
    def log_warning(cls, component: str, message: str):
        """Log a warning message."""
        if cls._initialized:
            logger = logging.getLogger(f'chainable.{component}')
            logger.warning(message)
    
    @classmethod
    def log_debug(cls, component: str, message: str):
        """Log a debug message."""
        if cls._initialized:
            logger = logging.getLogger(f'chainable.{component}')
            logger.debug(message)
    
    @classmethod
    def get_log_file_path(cls) -> Optional[Path]:
        """Get the current log file path."""
        return cls._log_file_path


class ProcessingError(Exception):
    """Custom exception for video processing errors."""
    def __init__(self, message: str, component: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.component = component
        self.details = details or {}


class ChainComponent(ABC):
    """Abstract base class for chainable video processing components."""
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.next_component: Optional['ChainComponent'] = None
        self.logger = self._setup_logger()
        
        # Ensure LogManager is initialized
        if not LogManager._initialized:
            LogManager.initialize()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for the component."""
        logger = logging.getLogger(f"chainable.{self.name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[{self.name}] %(levelname)s: %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def set_next(self, component: 'ChainComponent') -> 'ChainComponent':
        """Set the next component in the chain."""
        self.next_component = component
        return component
    
    @abstractmethod
    def process(self, data: VideoData) -> VideoData:
        """Process the video data. Must be implemented by subclasses."""
        pass
    
    def execute(self, data: VideoData) -> VideoData:
        """Execute this component and continue the chain."""
        try:
            # Validate input data
            self._validate_input(data)
            
            # Log processing start
            LogManager.log_info(self.name, f"Starting processing {data.frame_count} frames...")
            self.logger.info(f"Processing {data.frame_count} frames...")
            
            # Process the data
            processed_data = self.process(data)
            
            # Validate output data
            self._validate_output(processed_data)
            
            # Log processing completion
            LogManager.log_info(self.name, f"Processing completed successfully")
            self.logger.info("Processing completed successfully")
            
            # Continue the chain if there's a next component
            if self.next_component:
                return self.next_component.execute(processed_data)
            else:
                return processed_data
                
        except Exception as e:
            # Log error with full traceback
            error_msg = f"Processing failed: {str(e)}"
            LogManager.log_error(self.name, error_msg, e)
            self.logger.error(error_msg)
            
            # Re-raise as ProcessingError
            if isinstance(e, ProcessingError):
                raise
            else:
                raise ProcessingError(
                    error_msg,
                    component=self.name,
                    details={'original_exception': type(e).__name__}
                )
    
    def _validate_input(self, data: VideoData):
        """Validate input data. Override in subclasses for specific validation."""
        if not isinstance(data, VideoData):
            raise ProcessingError(
                f"Expected VideoData, got {type(data)}",
                component=self.name
            )
    
    def _validate_output(self, data: VideoData):
        """Validate output data. Override in subclasses for specific validation."""
        if not isinstance(data, VideoData):
            raise ProcessingError(
                f"Component produced invalid output: expected VideoData, got {type(data)}",
                component=self.name
            )


class ProgressReporter:
    """Simple progress reporter for processing operations."""
    
    def __init__(self, total_items: int, description: str = "Processing"):
        self.total_items = total_items
        self.current_item = 0
        self.description = description
        self.logger = logging.getLogger("chainable.progress")
    
    def update(self, increment: int = 1):
        """Update progress by increment."""
        self.current_item += increment
        if self.current_item > self.total_items:
            self.current_item = self.total_items
        
        percentage = (self.current_item / self.total_items) * 100
        if self.current_item % max(1, self.total_items // 10) == 0 or self.current_item == self.total_items:
            self.logger.info(f"{self.description}: {percentage:.1f}% ({self.current_item}/{self.total_items})")
    
    def finish(self):
        """Mark progress as finished."""
        self.current_item = self.total_items
        self.logger.info(f"{self.description}: Complete!")


class MemoryManager:
    """Memory management utilities for video processing."""
    
    @staticmethod
    def estimate_memory_usage(data: VideoData) -> Dict[str, float]:
        """Estimate memory usage for video data."""
        frame_size_bytes = data.frames.itemsize * np.prod(data.frames.shape[1:])
        total_bytes = frame_size_bytes * data.frame_count
        total_mb = total_bytes / (1024 * 1024)
        
        return {
            'frame_size_bytes': frame_size_bytes,
            'total_bytes': total_bytes,
            'total_mb': total_mb,
            'frame_count': data.frame_count
        }
    
    @staticmethod
    def should_use_batch_processing(data: VideoData, threshold_mb: float = 500.0) -> bool:
        """Determine if batch processing should be used based on memory usage."""
        memory_info = MemoryManager.estimate_memory_usage(data)
        return memory_info['total_mb'] > threshold_mb
    
    @staticmethod
    def calculate_optimal_batch_size(data: VideoData, target_mb: float = 100.0) -> int:
        """Calculate optimal batch size for processing."""
        memory_info = MemoryManager.estimate_memory_usage(data)
        frame_size_mb = memory_info['frame_size_bytes'] / (1024 * 1024)
        
        if frame_size_mb == 0:
            return min(10, data.frame_count)
        
        optimal_batch = max(1, int(target_mb / frame_size_mb))
        return min(optimal_batch, data.frame_count)
