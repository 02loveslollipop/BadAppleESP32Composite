import numpy as np
from numba import jit, prange
import math

@jit(nopython=True)
def lanczos_kernel(x, a=4):
    """Lanczos kernel function with parameter a (default 4 for Lanczos-4)."""
    if x == 0:
        return 1.0
    elif abs(x) < a:
        return a * math.sin(math.pi * x) * math.sin(math.pi * x / a) / (math.pi * math.pi * x * x)
    else:
        return 0.0

@jit(nopython=True, parallel=True)
def lanczos4_resize_cpu(image, new_width, new_height):
    """CPU-based Lanczos-4 interpolation resize using separable filtering (optimized)."""
    old_height, old_width = image.shape[:2]
    is_color = len(image.shape) == 3
    
    # First pass: horizontal resize
    if is_color:
        temp = np.zeros((old_height, new_width, image.shape[2]), dtype=np.float32)
    else:
        temp = np.zeros((old_height, new_width), dtype=np.float32)
    
    x_scale = old_width / new_width
    kernel_size = 4
    
    # Horizontal pass - process each column
    for x in prange(new_width):
        src_x = (x + 0.5) * x_scale - 0.5
        center_x = int(math.floor(src_x))
        
        # Calculate weights for this column
        weights = np.zeros(8, dtype=np.float32)  # -3 to +4 (8 positions)
        weight_sum = 0.0
        valid_samples = 0
        
        for kx in range(-kernel_size + 1, kernel_size + 1):
            sample_x = center_x + kx
            if 0 <= sample_x < old_width:
                weight = lanczos_kernel(src_x - sample_x, kernel_size)
                weights[kx + kernel_size - 1] = weight
                weight_sum += weight
                valid_samples += 1
        
        # Normalize weights
        if weight_sum > 0:
            for i in range(8):
                weights[i] /= weight_sum
        
        # Apply horizontal filtering
        for y in range(old_height):
            if is_color:
                for c in range(image.shape[2]):
                    pixel_value = 0.0
                    for kx in range(-kernel_size + 1, kernel_size + 1):
                        sample_x = center_x + kx
                        if 0 <= sample_x < old_width:
                            weight = weights[kx + kernel_size - 1]
                            pixel_value += image[y, sample_x, c] * weight
                    temp[y, x, c] = pixel_value
            else:
                pixel_value = 0.0
                for kx in range(-kernel_size + 1, kernel_size + 1):
                    sample_x = center_x + kx
                    if 0 <= sample_x < old_width:
                        weight = weights[kx + kernel_size - 1]
                        pixel_value += image[y, sample_x] * weight
                temp[y, x] = pixel_value
    
    # Second pass: vertical resize
    if is_color:
        resized = np.zeros((new_height, new_width, image.shape[2]), dtype=np.float32)
    else:
        resized = np.zeros((new_height, new_width), dtype=np.float32)
    
    y_scale = old_height / new_height
    
    # Vertical pass - process each row
    for y in prange(new_height):
        src_y = (y + 0.5) * y_scale - 0.5
        center_y = int(math.floor(src_y))
        
        # Calculate weights for this row
        weights = np.zeros(8, dtype=np.float32)
        weight_sum = 0.0
        
        for ky in range(-kernel_size + 1, kernel_size + 1):
            sample_y = center_y + ky
            if 0 <= sample_y < old_height:
                weight = lanczos_kernel(src_y - sample_y, kernel_size)
                weights[ky + kernel_size - 1] = weight
                weight_sum += weight
        
        # Normalize weights
        if weight_sum > 0:
            for i in range(8):
                weights[i] /= weight_sum
        
        # Apply vertical filtering
        for x in range(new_width):
            if is_color:
                for c in range(image.shape[2]):
                    pixel_value = 0.0
                    for ky in range(-kernel_size + 1, kernel_size + 1):
                        sample_y = center_y + ky
                        if 0 <= sample_y < old_height:
                            weight = weights[ky + kernel_size - 1]
                            pixel_value += temp[sample_y, x, c] * weight
                    resized[y, x, c] = pixel_value
            else:
                pixel_value = 0.0
                for ky in range(-kernel_size + 1, kernel_size + 1):
                    sample_y = center_y + ky
                    if 0 <= sample_y < old_height:
                        weight = weights[ky + kernel_size - 1]
                        pixel_value += temp[sample_y, x] * weight
                resized[y, x] = pixel_value
    
    # Convert back to original dtype and clamp values
    if image.dtype == np.uint8:
        resized = np.clip(resized, 0, 255).astype(np.uint8)
    elif image.dtype == np.uint16:
        resized = np.clip(resized, 0, 65535).astype(np.uint16)
    else:
        resized = resized.astype(image.dtype)
    
    return resized


__all__ = ['lanczos4_resize_cpu']