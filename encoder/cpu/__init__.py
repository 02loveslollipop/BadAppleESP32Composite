"""
CPU module for video processing algorithms.
"""

from .lanczos_resize import lanczos4_resize_cpu
from .frame_resampling import frame_resample_cpu


__all__ = ['lanczos4_resize_cpu', 'frame_resample_cpu']