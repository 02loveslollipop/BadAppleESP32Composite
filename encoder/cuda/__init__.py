"""
CUDA module for GPU-accelerated video processing.
"""

import warnings

# CUDA availability status
CUDA_AVAILABLE = False
cp = None

try:
    import cupy as _cp
    cp = _cp
    
    # Check if at least 1 CUDA device is available
    device_count = cp.cuda.runtime.getDeviceCount()
    if device_count > 0:
        CUDA_AVAILABLE = True
    else:
        warnings.warn("No CUDA devices found.")
        
except ImportError:
    warnings.warn("CuPy not installed. GPU acceleration unavailable.")
except Exception as e:
    warnings.warn(f"CUDA initialization failed: {str(e)}")

# Import GPU functions (will be no-ops if CUDA not available)
from .frame_resampling import frame_resample_gpu

__all__ = ['CUDA_AVAILABLE', 'cp', 'frame_resample_gpu']

