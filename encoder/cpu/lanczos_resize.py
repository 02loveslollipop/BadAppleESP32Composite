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
def _lanczos4_resize_cpu_horizontal_color(image, new_width):
    """CPU horizontal Lanczos-4 resize for color images using fixed 8-sample approach."""
    old_height, old_width, channels = image.shape
    temp = np.zeros((old_height, new_width, channels), dtype=np.float32)
    
    x_scale = old_width / new_width
    kernel_size = 4
    
    for x in prange(new_width):
        src_x = (x + 0.5) * x_scale - 0.5
        center_x = int(src_x)  # Match CUDA's astype(int32) behavior
        
        # Fixed 8-sample approach: [-3, -2, -1, 0, 1, 2, 3, 4] relative to center
        weights = np.zeros(8, dtype=np.float32)
        indices = np.zeros(8, dtype=np.int32)
        weight_sum = 0.0
        
        # Calculate weights for all 8 samples
        for i in range(8):
            kx = i - kernel_size + 1  # -3 to 4
            sample_x = center_x + kx
            
            # Clamp index to valid range
            clamped_x = max(0, min(sample_x, old_width - 1))
            indices[i] = clamped_x
            
            # Calculate weight only if sample is within bounds
            if 0 <= sample_x < old_width:
                weight = lanczos_kernel(src_x - sample_x, kernel_size)
                weights[i] = weight
                weight_sum += weight
            else:
                weights[i] = 0.0
        
        # Normalize weights
        if weight_sum > 0:
            for i in range(8):
                weights[i] /= weight_sum
        
        # Apply weights to all rows and channels
        for y in range(old_height):
            for c in range(channels):
                pixel_value = 0.0
                for i in range(8):
                    pixel_value += image[y, indices[i], c] * weights[i]
                temp[y, x, c] = pixel_value
    
    return temp

@jit(nopython=True, parallel=True)
def _lanczos4_resize_cpu_vertical_color(image, new_height):
    """CPU vertical Lanczos-4 resize for color images using fixed 8-sample approach."""
    old_height, old_width, channels = image.shape
    resized = np.zeros((new_height, old_width, channels), dtype=np.float32)
    
    y_scale = old_height / new_height
    kernel_size = 4
    
    for y in prange(new_height):
        src_y = (y + 0.5) * y_scale - 0.5
        center_y = int(src_y)  # Match CUDA's astype(int32) behavior
        
        # Fixed 8-sample approach: [-3, -2, -1, 0, 1, 2, 3, 4] relative to center
        weights = np.zeros(8, dtype=np.float32)
        indices = np.zeros(8, dtype=np.int32)
        weight_sum = 0.0
        
        # Calculate weights for all 8 samples
        for i in range(8):
            ky = i - kernel_size + 1  # -3 to 4
            sample_y = center_y + ky
            
            # Clamp index to valid range
            clamped_y = max(0, min(sample_y, old_height - 1))
            indices[i] = clamped_y
            
            # Calculate weight only if sample is within bounds
            if 0 <= sample_y < old_height:
                weight = lanczos_kernel(src_y - sample_y, kernel_size)
                weights[i] = weight
                weight_sum += weight
            else:
                weights[i] = 0.0
        
        # Normalize weights
        if weight_sum > 0:
            for i in range(8):
                weights[i] /= weight_sum
        
        # Apply weights to all columns and channels
        for x in range(old_width):
            for c in range(channels):
                pixel_value = 0.0
                for i in range(8):
                    pixel_value += image[indices[i], x, c] * weights[i]
                resized[y, x, c] = pixel_value
    
    return resized

@jit(nopython=True, parallel=True)
def _lanczos4_resize_cpu_horizontal_gray(image, new_width):
    """CPU horizontal Lanczos-4 resize for grayscale images using fixed 8-sample approach."""
    old_height, old_width = image.shape
    temp = np.zeros((old_height, new_width), dtype=np.float32)
    
    x_scale = old_width / new_width
    kernel_size = 4
    
    for x in prange(new_width):
        src_x = (x + 0.5) * x_scale - 0.5
        center_x = int(src_x)  # Match CUDA's astype(int32) behavior
        
        # Fixed 8-sample approach: [-3, -2, -1, 0, 1, 2, 3, 4] relative to center
        weights = np.zeros(8, dtype=np.float32)
        indices = np.zeros(8, dtype=np.int32)
        weight_sum = 0.0
        
        # Calculate weights for all 8 samples
        for i in range(8):
            kx = i - kernel_size + 1  # -3 to 4
            sample_x = center_x + kx
            
            # Clamp index to valid range
            clamped_x = max(0, min(sample_x, old_width - 1))
            indices[i] = clamped_x
            
            # Calculate weight only if sample is within bounds
            if 0 <= sample_x < old_width:
                weight = lanczos_kernel(src_x - sample_x, kernel_size)
                weights[i] = weight
                weight_sum += weight
            else:
                weights[i] = 0.0
        
        # Normalize weights
        if weight_sum > 0:
            for i in range(8):
                weights[i] /= weight_sum
        
        # Apply weights to all rows
        for y in range(old_height):
            pixel_value = 0.0
            for i in range(8):
                pixel_value += image[y, indices[i]] * weights[i]
            temp[y, x] = pixel_value
    
    return temp

@jit(nopython=True, parallel=True)
def _lanczos4_resize_cpu_vertical_gray(image, new_height):
    """CPU vertical Lanczos-4 resize for grayscale images using fixed 8-sample approach."""
    old_height, old_width = image.shape
    resized = np.zeros((new_height, old_width), dtype=np.float32)
    
    y_scale = old_height / new_height
    kernel_size = 4
    
    for y in prange(new_height):
        src_y = (y + 0.5) * y_scale - 0.5
        center_y = int(src_y)  # Match CUDA's astype(int32) behavior
        
        # Fixed 8-sample approach: [-3, -2, -1, 0, 1, 2, 3, 4] relative to center
        weights = np.zeros(8, dtype=np.float32)
        indices = np.zeros(8, dtype=np.int32)
        weight_sum = 0.0
        
        # Calculate weights for all 8 samples
        for i in range(8):
            ky = i - kernel_size + 1  # -3 to 4
            sample_y = center_y + ky
            
            # Clamp index to valid range
            clamped_y = max(0, min(sample_y, old_height - 1))
            indices[i] = clamped_y
            
            # Calculate weight only if sample is within bounds
            if 0 <= sample_y < old_height:
                weight = lanczos_kernel(src_y - sample_y, kernel_size)
                weights[i] = weight
                weight_sum += weight
            else:
                weights[i] = 0.0
        
        # Normalize weights
        if weight_sum > 0:
            for i in range(8):
                weights[i] /= weight_sum
        
        # Apply weights to all columns
        for x in range(old_width):
            pixel_value = 0.0
            for i in range(8):
                pixel_value += image[indices[i], x] * weights[i]
            resized[y, x] = pixel_value
    
    return resized

def lanczos4_resize_cpu_color(image, new_width, new_height):
    """CPU-based Lanczos-4 interpolation resize for color images."""
    # Convert to float32 for processing
    image_float = image.astype(np.float32)
    
    # First pass: horizontal resize
    temp = _lanczos4_resize_cpu_horizontal_color(image_float, new_width)
    
    # Second pass: vertical resize
    resized = _lanczos4_resize_cpu_vertical_color(temp, new_height)
    
    # Convert back to original dtype with clamping
    dtype_str = str(image.dtype)
    if 'uint8' in dtype_str:
        result = np.clip(resized, 0, 255).astype(np.uint8)
    elif 'uint16' in dtype_str:
        result = np.clip(resized, 0, 65535).astype(np.uint16)
    elif 'float32' in dtype_str:
        result = resized.astype(np.float32)
    elif 'float64' in dtype_str:
        result = resized.astype(np.float64)
    else:
        result = np.clip(resized, 0, 255).astype(np.uint8)
    
    return result

def lanczos4_resize_cpu_gray(image, new_width, new_height):
    """CPU-based Lanczos-4 interpolation resize for grayscale images."""
    # Convert to float32 for processing
    image_float = image.astype(np.float32)
    
    # First pass: horizontal resize
    temp = _lanczos4_resize_cpu_horizontal_gray(image_float, new_width)
    
    # Second pass: vertical resize
    resized = _lanczos4_resize_cpu_vertical_gray(temp, new_height)
    
    # Convert back to original dtype with clamping
    dtype_str = str(image.dtype)
    if 'uint8' in dtype_str:
        result = np.clip(resized, 0, 255).astype(np.uint8)
    elif 'uint16' in dtype_str:
        result = np.clip(resized, 0, 65535).astype(np.uint16)
    elif 'float32' in dtype_str:
        result = resized.astype(np.float32)
    elif 'float64' in dtype_str:
        result = resized.astype(np.float64)
    else:
        result = np.clip(resized, 0, 255).astype(np.uint8)
    
    return result

def lanczos4_resize_cpu(image, new_width, new_height):
    """CPU-based Lanczos-4 interpolation resize dispatcher."""
    if len(image.shape) == 3:
        return lanczos4_resize_cpu_color(image, new_width, new_height)
    else:
        return lanczos4_resize_cpu_gray(image, new_width, new_height)

def lanczos_resize_cpu(image, new_width, new_height, kernel_size=4):
    """
    CPU-based Lanczos interpolation resize with configurable kernel size.
    
    Args:
        image: Input image (numpy array)
        new_width: Target width
        new_height: Target height
        kernel_size: Lanczos kernel size (2, 3, 4, etc.)
    
    Returns:
        Resized image as numpy array
    """
    if kernel_size == 4:
        # Use optimized Lanczos-4 implementation
        return lanczos4_resize_cpu(image, new_width, new_height)
    else:
        # For other kernel sizes, use fallback or raise error
        raise NotImplementedError(f"CPU Lanczos resize with kernel_size={kernel_size} not yet implemented. Only kernel_size=4 is supported.")

__all__ = ['lanczos4_resize_cpu', 'lanczos_resize_cpu']

