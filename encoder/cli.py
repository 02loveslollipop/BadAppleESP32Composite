import click
import cv2
import numpy as np
from pathlib import Path

# Import resize functions
from .cpu.lanczos_resize import lanczos4_resize_cpu
try:
    from .cuda.lanczos_resize import lanczos4_resize_gpu
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    def lanczos4_resize_gpu(*args, **kwargs):
        raise RuntimeError("CUDA not available. Install CuPy for GPU acceleration.")

 

class Resolution(click.ParamType):
    name = 'resolution'

    def convert(self, value, param, ctx):
        if value is None:
            return None
        try:
            if 'x' not in value:
                self.fail(f"Invalid resolution format: {value}. Use 'widthxheight' format.", param, ctx)
            width, height = value.split('x')
            return int(width), int(height)
        except ValueError:
            self.fail(f'{value} is not a valid resolution format. Use WIDTHxHEIGHT (e.g., 120x75)', param, ctx)
    
class ResampleTechnique(click.Choice):
    def __init__(self) -> None:
        super().__init__(['lanczos', 'nearest', 'bicubic', 'linear', 'bits2', 'area'])
    
    def convert(self, value, param, ctx):
        value = super().convert(value, param, ctx)
        if value is None:
            return None
        if isinstance(value, str) and value.lower() not in self.choices:
            self.fail(f'Invalid resample technique: {value}. Choose from {", ".join(self.choices)}.', param, ctx)
        return value.lower() if isinstance(value, str) else value

class CudaDevice(click.ParamType):
    name = 'cuda_device'

    def convert(self, value, param, ctx):
        if value is None:
            return None
        try:
            device_id = int(value)
            if device_id < 0:
                self.fail(f'Invalid CUDA device ID: {value}. Must be a non-negative integer.', param, ctx)
            return device_id
        except ValueError:
            self.fail(f'{value} is not a valid CUDA device ID. Must be a non-negative integer.', param, ctx)

@click.command()
@click.option('--input', '-i', type=click.Path(exists=True, dir_okay=False), required=True, help='Input video file path.')
@click.option('-r', '--resolution', type=Resolution(), help='Set the resolution of the video (e.g., 120x75). Default is video resolution.')
@click.option('-t', '--threshold', type=int, default=60, help='Set the threshold for black/white conversion. Default is 60.')
@click.option('-rs', '--resample', type=ResampleTechnique(), default='lanczos', help='Set the resampling technique. Default is lanczos.')
@click.option('--dt', '--dither', is_flag=True, default=False, help='Enable dithering for the output video.')
@click.option('--interactive', '-it', is_flag=True, default=False, help='Run in interactive mode.')

@click.option('--cuda', '-c', type=CudaDevice(), help='Enable CUDA support with device ID (e.g., 0 for first GPU). Requires CUDA GPU support.')
@click.option('--cuda-device', type=CudaDevice(), help='Specify CUDA device ID (0, 1, 2...). Only used with --cuda flag.')

# the following options can only be mixed in an specific way:

# RLE: Run Length Encoding, used for compressing the video by encoding sequences of repeated deltas in a line.
# MC: Motion Compensation, used for compressing the video by computing the movement of pixel groups between frames.

# both options can be used together.

# scanline: This option only output half of the vertical resolution and leaves the other half black. this option can be used with RLE but not with MC.

# interlaced: This option outputs the video in interlaced format, which is not compatible with RLE or MC. In this option each frame only half of the vertical resolution is updated, the other half is left unchanged. This option can't be used with scanline.

# so:

# RLE + MC: valid

# SCANLINE + (compression technique): valid
# INTERLACED + (compression technique): not valid
# INTERLACED + SCANLINE: not valid

@click.option('--rle', is_flag=True, default=False, help='Enable Run Length Encoding for compression.')
@click.option('--mc', is_flag=True, default=False, help='Enable Motion Compensation for compression.')
@click.option('--scanline', is_flag=True, default=False, help='Enable scanline mode (half vertical resolution).')
@click.option('--interlaced', is_flag=True, default=False, help='Enable interlaced mode (half vertical resolution per frame).')
@click.pass_context
def main(ctx, input, resolution, threshold, resample, dither, interactive, cuda, cuda_device, rle, mc, scanline, interlaced):
    """CLI for video and audio encoding for ESP32 (specifically for bad apple)."""
    # Validate parameter combinations
    if interlaced and (rle or mc):
        raise click.BadParameter('Interlaced mode cannot be used with RLE or MC compression techniques.')
    
    if scanline and interlaced:
        raise click.BadParameter('Scanline mode cannot be used with interlaced mode.')

    # Validate CUDA options
    if cuda is not None and not CUDA_AVAILABLE:
        raise click.BadParameter('CUDA support requested but not available. Install CuPy for GPU acceleration.')
    
    if cuda_device is not None and cuda is None:
        raise click.BadParameter('--cuda-device specified but --cuda not enabled. Use --cuda <device_id> instead.')

    # Determine processing device
    use_cuda = cuda is not None
    device_id = cuda if cuda is not None else (cuda_device if cuda_device is not None else 0)
    
    # Determine resize function based on resample technique and device
    if resample == 'lanczos':
        if use_cuda:
            resize_func = lanczos4_resize_gpu
            click.echo(f'🚀 Using CUDA Lanczos-4 on device {device_id}')
        else:
            resize_func = lanczos4_resize_cpu
            click.echo('🖥️  Using CPU Lanczos-4')
    else:
        # For other techniques, use OpenCV (CPU only for now)
        resize_func = None
        use_cuda = False
        click.echo(f'🖥️  Using OpenCV {resample} (CPU)')
    
    # Process the video
    process_video(
        input_path=input,
        output_resolution=resolution,
        threshold=threshold,
        resample_technique=resample,
        resize_func=resize_func,
        use_cuda=use_cuda,
        device_id=device_id,
        dither=dither,
        rle=rle,
        mc=mc,
        scanline=scanline,
        interlaced=interlaced,        interactive=interactive
    )

def process_video(input_path, output_resolution, threshold, resample_technique, resize_func, 
                 use_cuda, device_id, dither, rle, mc, scanline, interlaced, interactive):
    """Process the video with the specified parameters."""
    click.echo(f'🎬 Processing video: {input_path}')
    
    # Open video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise click.ClickException(f"Cannot open video file: {input_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    click.echo(f"📹 Video info: {original_width}x{original_height}, {fps:.2f} fps, {frame_count} frames")
    
    # Determine output resolution
    if output_resolution:
        target_width, target_height = output_resolution
    else:
        target_width, target_height = original_width, original_height
    
    click.echo(f"🎯 Target resolution: {target_width}x{target_height}")
    
    # Adjust for scanline mode
    if scanline:
        target_height = target_height // 2
        click.echo(f"📺 Scanline mode: effective resolution {target_width}x{target_height}")
    
    # Process frames
    frame_num = 0
    with click.progressbar(length=frame_count, label='Processing frames') as bar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame if needed
            if (target_width != original_width or target_height != original_height):
                if resize_func:
                    # Use custom resize function (Lanczos)
                    if use_cuda:
                        # Convert to GPU array if using CUDA
                        import cupy as cp
                        with cp.cuda.Device(device_id):
                            frame_resized = resize_func(frame, target_width, target_height)
                            if isinstance(frame_resized, cp.ndarray):
                                frame_resized = cp.asnumpy(frame_resized)
                    else:
                        frame_resized = resize_func(frame, target_width, target_height)
                else:
                    # Use OpenCV resize
                    interpolation_map = {
                        'nearest': cv2.INTER_NEAREST,
                        'linear': cv2.INTER_LINEAR,
                        'bicubic': cv2.INTER_CUBIC,
                        'area': cv2.INTER_AREA,
                        'bits2': cv2.INTER_LANCZOS4  # Fallback to Lanczos4
                    }
                    interp = interpolation_map.get(resample_technique, cv2.INTER_LANCZOS4)
                    frame_resized = cv2.resize(frame, (target_width, target_height), interpolation=interp)
            else:
                frame_resized = frame
            
            # TODO: Add additional processing here:
            # - Threshold conversion
            # - Dithering
            # - RLE compression
            # - Motion compensation
            # - Scanline/interlaced handling
            
            frame_num += 1
            bar.update(1)
            
            # For now, just process a few frames in interactive mode
            if interactive and frame_num >= 10:
                click.echo("\n🔄 Interactive mode: processed 10 frames as demo")
                break
    
    cap.release()
    click.echo(f"\n✅ Processing complete! Processed {frame_num} frames")
    
if __name__ == '__main__':
    main()