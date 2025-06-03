"""
Chainable video processing components.

This package provides a modular, chainable video processing architecture where each 
component has a single responsibility and can be linked together to form a complete 
video encoding pipeline.
"""

from .basex import VideoData, ChainComponent, ProcessingError, ProgressReporter, MemoryManager, LogManager
from .openx import VideoOpener, open_video
from .resizex import VideoResizer, resize_video

__all__ = [
    # Base classes
    'VideoData',
    'ChainComponent', 
    'ProcessingError',
    'ProgressReporter',
    'MemoryManager',
    'LogManager',
    
    # Components
    'VideoOpener',
    'VideoResizer',
    
    # Convenience functions
    'open_video',
    'resize_video'
]
