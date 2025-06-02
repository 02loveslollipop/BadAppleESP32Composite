"""
CUDA-accelerated Lanczos resizing using CuPy.
"""

from . import CUDA_AVAILABLE, cp

if not CUDA_AVAILABLE:
    def lanczos4_resize_gpu(*args, **kwargs):
        raise RuntimeError("CUDA not available. No CUDA devices found or CuPy not installed.")
        
else:
    def lanczos_kernel_gpu(x, a=4):
        """GPU Lanczos kernel function."""
        return cp.where(
            x == 0, 
            1.0,
            cp.where(
                cp.abs(x) < a,
                a * cp.sin(cp.pi * x) * cp.sin(cp.pi * x / a) / (cp.pi * cp.pi * x * x),
                0.0
            )
        )

    def lanczos4_resize_gpu(image, new_width, new_height):
        """Optimized GPU Lanczos-4 using separable filtering."""
        if not isinstance(image, cp.ndarray):
            image = cp.asarray(image, dtype=cp.float32)
        else:
            image = image.astype(cp.float32)
        
        old_height, old_width = image.shape[:2]
        is_color = len(image.shape) == 3
        
        # First pass: horizontal resize
        if is_color:
            temp = cp.zeros((old_height, new_width, image.shape[2]), dtype=cp.float32)
        else:
            temp = cp.zeros((old_height, new_width), dtype=cp.float32)
        
        x_scale = old_width / new_width
        kernel_size = 4
        
        # Horizontal pass
        for x in range(new_width):
            src_x = (x + 0.5) * x_scale - 0.5
            center_x = int(cp.floor(src_x))
            
            weight_sum = 0.0
            for kx in range(-kernel_size + 1, kernel_size + 1):
                sample_x = center_x + kx
                if 0 <= sample_x < old_width:
                    weight = lanczos_kernel_gpu(src_x - sample_x, kernel_size)
                    weight_sum += weight
                    if is_color:
                        temp[:, x, :] += image[:, sample_x, :] * weight
                    else:
                        temp[:, x] += image[:, sample_x] * weight
            
            # Normalize horizontal pass
            if weight_sum > 0:
                if is_color:
                    temp[:, x, :] /= weight_sum
                else:
                    temp[:, x] /= weight_sum
        
        # Second pass: vertical resize
        if is_color:
            resized = cp.zeros((new_height, new_width, image.shape[2]), dtype=cp.float32)
        else:
            resized = cp.zeros((new_height, new_width), dtype=cp.float32)
        
        y_scale = old_height / new_height
        
        # Vertical pass
        for y in range(new_height):
            src_y = (y + 0.5) * y_scale - 0.5
            center_y = int(cp.floor(src_y))
            
            weight_sum = 0.0
            for ky in range(-kernel_size + 1, kernel_size + 1):
                sample_y = center_y + ky
                if 0 <= sample_y < old_height:
                    weight = lanczos_kernel_gpu(src_y - sample_y, kernel_size)
                    weight_sum += weight
                    if is_color:
                        resized[y, :, :] += temp[sample_y, :, :] * weight
                    else:
                        resized[y, :] += temp[sample_y, :] * weight
            
            # Normalize vertical pass
            if weight_sum > 0:
                if is_color:
                    resized[y, :, :] /= weight_sum
                else:
                    resized[y, :] /= weight_sum
        
        # Convert back to original dtype
        if image.dtype == cp.uint8:
            resized = cp.clip(resized, 0, 255).astype(cp.uint8)
        elif image.dtype == cp.uint16:
            resized = cp.clip(resized, 0, 65535).astype(cp.uint16)
        
        return resized

__all__ = ['lanczos4_resize_gpu']