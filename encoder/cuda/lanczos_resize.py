from . import CUDA_AVAILABLE, cp

if not CUDA_AVAILABLE:
    def lanczos4_resize_gpu(*args, **kwargs):
        raise RuntimeError("CUDA not available. No CUDA devices found or CuPy not installed.")
        
else:
    def lanczos_kernel_gpu(x, a=4):
        """Generates Lanczos kernel values on GPU."""
        x = cp.asarray(x)
        return cp.where(
            x == 0, 
            1.0,
            cp.where(
                cp.abs(x) < a,
                a * cp.sin(cp.pi * x) * cp.sin(cp.pi * x / a) / (cp.pi * cp.pi * x * x),
                0.0
            )
        )

    def _lanczos4_resize_gpu_horizontal(image, new_width):
        """Horizontal resize using Lanczos kernel on GPU."""
        old_height, old_width = image.shape[:2]
        is_color = len(image.shape) == 3
        
        x_scale = old_width / new_width
        kernel_size = 4
          # Vectorized coordinate calculation for all output pixels
        x_coords = cp.arange(new_width, dtype=cp.float32)
        src_x = (x_coords + 0.5) * x_scale - 0.5
        center_x = src_x.astype(cp.int32)  # Changed from cp.floor to match CPU int() behavior
        
        # Create indices for all kernel samples for all output pixels
        sample_offsets = cp.arange(-kernel_size + 1, kernel_size + 1)  # [-3, -2, -1, 0, 1, 2, 3, 4]
        
        # Broadcast to create all sample indices: [new_width, 8]
        all_sample_indices = center_x[:, cp.newaxis] + sample_offsets[cp.newaxis, :]
        
        # Create valid masks
        valid_mask = (all_sample_indices >= 0) & (all_sample_indices < old_width)
        
        # Calculate distances and weights for all samples
        distances = src_x[:, cp.newaxis] - all_sample_indices
        weights = lanczos_kernel_gpu(distances, kernel_size)
        
        # Zero out weights for invalid samples
        weights = cp.where(valid_mask, weights, 0.0)
        
        # Normalize weights for each output pixel
        weight_sums = cp.sum(weights, axis=1, keepdims=True)
        weight_sums = cp.where(weight_sums > 0, weight_sums, 1.0)
        weights = weights / weight_sums
        
        # Clamp indices to valid range for gathering
        clamped_indices = cp.clip(all_sample_indices, 0, old_width - 1)
        
        if is_color:
            # Create output array
            output = cp.zeros((old_height, new_width, image.shape[2]), dtype=cp.float32)
            
            # Vectorized gather and weighted sum for color images
            for h in range(old_height):
                # Shape: [new_width, 8, 3]
                gathered_pixels = image[h, clamped_indices, :]
                # Shape: [new_width, 3]
                output[h, :, :] = cp.sum(gathered_pixels * weights[:, :, cp.newaxis], axis=1)
        else:
            # Create output array
            output = cp.zeros((old_height, new_width), dtype=cp.float32)
            
            # Vectorized gather and weighted sum for grayscale
            for h in range(old_height):
                # Shape: [new_width, 8]
                gathered_pixels = image[h, clamped_indices]
                # Shape: [new_width]
                output[h, :] = cp.sum(gathered_pixels * weights, axis=1)
        
        return output

    def _lanczos4_resize_gpu_vertical(image, new_height):
        """Vertical resize using Lanczos kernel on GPU."""
        old_height, old_width = image.shape[:2]
        is_color = len(image.shape) == 3
        
        y_scale = old_height / new_height
        kernel_size = 4
          # Vectorized coordinate calculation for all output pixels
        y_coords = cp.arange(new_height, dtype=cp.float32)
        src_y = (y_coords + 0.5) * y_scale - 0.5
        center_y = src_y.astype(cp.int32)  # Changed from cp.floor to match CPU int() behavior
        
        # Create indices for all kernel samples for all output pixels
        sample_offsets = cp.arange(-kernel_size + 1, kernel_size + 1)  # [-3, -2, -1, 0, 1, 2, 3, 4]
        
        # Broadcast to create all sample indices: [new_height, 8]
        all_sample_indices = center_y[:, cp.newaxis] + sample_offsets[cp.newaxis, :]
        
        # Create valid masks
        valid_mask = (all_sample_indices >= 0) & (all_sample_indices < old_height)
        
        # Calculate distances and weights for all samples
        distances = src_y[:, cp.newaxis] - all_sample_indices
        weights = lanczos_kernel_gpu(distances, kernel_size)
        
        # Zero out weights for invalid samples
        weights = cp.where(valid_mask, weights, 0.0)
        
        # Normalize weights for each output pixel
        weight_sums = cp.sum(weights, axis=1, keepdims=True)
        weight_sums = cp.where(weight_sums > 0, weight_sums, 1.0)
        weights = weights / weight_sums
        
        # Clamp indices to valid range for gathering
        clamped_indices = cp.clip(all_sample_indices, 0, old_height - 1)
        
        if is_color:
            # Create output array
            output = cp.zeros((new_height, old_width, image.shape[2]), dtype=cp.float32)
            
            # Vectorized gather and weighted sum for color images
            for w in range(old_width):
                # Shape: [new_height, 8, 3]
                gathered_pixels = image[clamped_indices, w, :]
                # Shape: [new_height, 3]
                output[:, w, :] = cp.sum(gathered_pixels * weights[:, :, cp.newaxis], axis=1)
        else:
            # Create output array
            output = cp.zeros((new_height, old_width), dtype=cp.float32)
            
            # Vectorized gather and weighted sum for grayscale
            for w in range(old_width):
                # Shape: [new_height, 8]
                gathered_pixels = image[clamped_indices, w]
                # Shape: [new_height]
                output[:, w] = cp.sum(gathered_pixels * weights, axis=1)
        
        return output

    def lanczos4_resize_gpu(image, new_width, new_height):
        """CUDA optimized Lanczos-4 resize function."""
       
        original_dtype = image.dtype # Save the dtype to restore later if GPU operations change it
        
        
        if not isinstance(image, cp.ndarray): #Ensure is CuPy array if not already
            image = cp.asarray(image, dtype=cp.float32)
        else:
            image = image.astype(cp.float32) # Convert to float32 as GPU operations require floating point
        
        temp = _lanczos4_resize_gpu_horizontal(image, new_width) #Resize to target horizontal size
        
        resized = _lanczos4_resize_gpu_vertical(temp, new_height) #Resize to target vertical size
        
        # Convert type back to the one of the original variable
        dtype_str = str(original_dtype)        
        if 'uint8' in dtype_str:
            result = cp.clip(resized, 0, 255).astype(cp.uint8)
        elif 'uint16' in dtype_str:
            result = cp.clip(resized, 0, 65535).astype(cp.uint16)
        elif 'float32' in dtype_str:
            result = resized.astype(cp.float32)
        elif 'float64' in dtype_str:
            result = resized.astype(cp.float64)
        else:
            result = cp.clip(resized, 0, 255).astype(cp.uint8)
        
        return result

    def lanczos_resize_gpu(image, new_width, new_height, kernel_size=4):
        """CUDA optimized general Lanczos resize function."""
        
        if kernel_size == 4: #if kernel size is 4, use optimized Lanczos-4 implementation
            # Use optimized Lanczos-4 implementation
            return lanczos4_resize_gpu(image, new_width, new_height)
        else:
            # For other kernel sizes, use fallback or raise error
            raise NotImplementedError(f"GPU Lanczos resize with kernel_size={kernel_size} not yet implemented. Only kernel_size=4 is supported.")

__all__ = ['lanczos4_resize_gpu', 'lanczos_resize_gpu'] if CUDA_AVAILABLE else []