import click
import numpy as np
from pathlib import Path
import time
from PIL import Image
import traceback #TODO: remove this import, it is only used for debugging

# Import resize functions - supports configurable Lanczos kernel sizes
from .cpu.lanczos_resize import lanczos4_resize_cpu, lanczos_resize_cpu
try:
    from .cuda.lanczos_resize import lanczos4_resize_gpu, lanczos_resize_gpu
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    def lanczos4_resize_gpu(*args, **kwargs):
        raise RuntimeError("CUDA not available. Install CuPy for GPU acceleration.")
    def lanczos_resize_gpu(*args, **kwargs):
        raise RuntimeError("CUDA not available. Install CuPy for GPU acceleration.")

# Import task scheduler and video utilities
from .video_scheduler import VideoResizeTaskScheduler, VideoValidator, VideoFrameExtractor, export_frames_as_numpy_array, save_resized_frames_to_video


 

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
    
class CudaDevice(click.ParamType):
    name = 'cuda'

    def convert(self, value, param, ctx):
        if value is None:
            return None
        if value.lower() == 'auto':
            return 0  # Default to first GPU
        try:
            device_id = int(value)
            if device_id < 0:
                self.fail(f'Invalid CUDA device ID: {value}. Must be a non-negative integer or "auto".', param, ctx)
            return device_id
        except ValueError:
            self.fail(f'{value} is not a valid CUDA device ID. Must be a non-negative integer or "auto".', param, ctx)

@click.command()
@click.option('--input', '-i', type=click.Path(exists=True, dir_okay=False), required=True, help='Input video file path.')
@click.option('-r', '--resolution', type=Resolution(), help='Set the resolution of the video (e.g., 120x75). Default is video resolution.')
@click.option('-t', '--threshold', type=int, default=60, help='Set the threshold for black/white conversion. Default is 60.')
@click.option('-lc', '--lanczos', type=int, default=None, help='Lanczos kernel size (2, 3, 4, etc.). Auto-enabled with --resolution if not specified.')
@click.option('--dither', '--dt', is_flag=True, default=False, help='Enable dithering for the output video.')
@click.option('--interactive', '-it', is_flag=True, default=False, help='Run in interactive mode.')
@click.option('--cuda', '-c', type=CudaDevice(), help='Enable CUDA acceleration. Specify device ID (0, 1, 2, etc.) or "auto" for automatic GPU selection. Requires CUDA-compatible GPU and CuPy.')
@click.option('--workers', '-w', type=int, default=4, help='Number of worker threads for parallel processing. Default is 4.')
@click.option('-e', '--execution-mode', type=click.Choice(['threading', 'multiprocessing']), default='threading', help='Execution mode for parallel processing. Default is threading.')
@click.option('-b', '--batch-size', type=int, default=50, help='Batch size for frame processing. Default is 50 (increased for better GPU utilization).')
@click.option('--output-format', type=click.Choice(['array', 'frames', 'video']), default='array',
              help='Output format: array (numpy), frames (list), or video (file). Default is array.')
@click.option('--output-path', '-o', type=click.Path(), help='Output path for processed video or data.')

# Frame rate conversion options
@click.option('-f','--target-fps', type=float, help='Target frame rate for video resampling (e.g., 15.0, 30.0). Must be lower than source FPS.')
@click.option('--reframe-technique', type=click.Choice(['simple', 'intelligent']), default='intelligent',
              help='Frame resampling technique: simple (ratio-based) or intelligent (quality-based). Default is intelligent.')

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

@click.option('--rle', '--run-length', is_flag=True, default=False, help='Enable Run Length Encoding for compression.')
@click.option('--mc', is_flag=True, default=False, help='Enable Motion Compensation for compression.')
@click.option('--scanline', is_flag=True, default=False, help='Enable scanline mode (half vertical resolution).')
@click.option('--interlaced', is_flag=True, default=False, help='Enable interlaced mode (half vertical resolution per frame).')
@click.pass_context
def main(ctx, input, resolution, threshold, lanczos, dither, interactive, cuda, workers, execution_mode, batch_size, output_format, output_path, target_fps, reframe_technique, rle, mc, scanline, interlaced):
    """CLI for video and audio encoding for ESP32 (specifically for bad apple)."""
    # Validate parameter combinations
    if interlaced and (rle or mc):
        raise click.BadParameter('Interlaced mode cannot be used with RLE or MC compression techniques.')
    
    if scanline and interlaced:
        raise click.BadParameter('Scanline mode cannot be used with interlaced mode.')    # Validate CUDA options
    if cuda is not None and not CUDA_AVAILABLE:
        raise click.BadParameter('CUDA support requested but not available. Install CuPy for GPU acceleration.')
    
    # Validate output options
    if output_format == 'video' and not output_path:
        raise click.BadParameter('Output path is required when output format is "video".')
    
    # Validate frame rate options
    if target_fps is not None:
        if target_fps <= 0:
            raise click.BadParameter('Target FPS must be positive.')
      # Determine processing device
    use_cuda = cuda is not None
    device_id = cuda if cuda is not None else 0
    
    # Handle resize logic: auto-enable Lanczos when resolution is set
    needs_resize = resolution is not None
    
    if needs_resize:
        # Auto-enable Lanczos if not specified
        if lanczos is None:
            lanczos = 4  # Default to Lanczos-4
            click.echo('Auto-enabled Lanczos-4 resize for resolution change')
        
        # Validate kernel size
        if lanczos < 2:
            raise click.BadParameter('Lanczos kernel size must be at least 2.')
          # Determine resize function
        if use_cuda:
            if lanczos == 4:
                resize_func = lanczos4_resize_gpu
                click.echo(f'Using CUDA Lanczos-4 on device {device_id}')
            else:
                resize_func = lanczos_resize_gpu
                click.echo(f'Using CUDA Lanczos-{lanczos} on device {device_id}')
        else:
            if lanczos == 4:
                resize_func = lanczos4_resize_cpu
                click.echo('Using CPU Lanczos-4')
            else:
                resize_func = lanczos_resize_cpu
                click.echo(f'Using CPU Lanczos-{lanczos}')
    else:        # No resize needed
        if lanczos is not None:
            click.echo('Warning: --lanczos specified but no --resolution set. Resize will be skipped.')
        resize_func = None

    # Process the video using chainable components
    try:
        from .chainable import VideoOpener, VideoResizer, VideoReframer, VideoTemporal, LogManager
        
        # Initialize comprehensive logging system for processing run
        LogManager.initialize()
        click.echo(f'Logging initialized: {LogManager.get_log_file_path()}')
        
        # Initialize the processing chain
        # Step 1: Open video file
        opener = VideoOpener(
            file_path=input,
            color_mode='RGB',  # Default to RGB for now
            max_frames=None   # Extract all frames
        )
        
        click.echo(f'Opening video file: {input}')
        video_data = opener.open()
        
        click.echo(f'Loaded video: {video_data.frame_count} frames, '
                  f'{video_data.resolution[0]}x{video_data.resolution[1]}, '
                  f'{video_data.frame_rate:.2f} fps, {video_data.color_mode}')
          # Step 2: Resize if needed
        if needs_resize:
            resizer = VideoResizer(
                target_resolution=resolution,
                lanczos_kernel=lanczos,
                use_gpu=use_cuda,
                gpu_device=device_id,
                workers=workers,
                execution_mode=execution_mode,
                batch_size=batch_size
            )
            
            click.echo(f'Resizing to {resolution[0]}x{resolution[1]} using Lanczos-{lanczos}')
            video_data = resizer.process(video_data)
            
            click.echo(f'Resize complete: {video_data.resolution[0]}x{video_data.resolution[1]}')
        
        # Step 3: Frame rate conversion if needed
        if target_fps is not None:
            if target_fps >= video_data.frame_rate:
                click.echo(f'Warning: Target FPS ({target_fps:.2f}) is not lower than source FPS ({video_data.frame_rate:.2f}). Skipping reframe.')
            else:
                reframer = VideoReframer(
                    target_fps=target_fps,
                    technique=reframe_technique,
                    use_gpu=use_cuda,
                    gpu_device=device_id
                )
                click.echo(f'Converting frame rate from {video_data.frame_rate:.2f} to {target_fps:.2f} fps using {reframe_technique} technique')
                video_data = reframer.process(video_data)
                
                click.echo(f'Frame rate conversion complete: {video_data.frame_count} frames at {video_data.frame_rate:.2f} fps')
          # Step 4: Temporal display (TODO: Remove this chainable before production - development only)
        temporal = VideoTemporal(
            playback_fps=None,  # Use video's native FPS
            window_title="Video Processing Preview - DEVELOPMENT MODE",
            show_info=True,
            show_controls=True,
            auto_play=True,
            loop=True
        )
        
        click.echo('Opening temporal display: Video Processing Preview - DEVELOPMENT MODE')
        click.echo('Controls: SPACE=play/pause, R=reset, LEFT/RIGHT=step, ESC=close')
        click.echo('TODO: This temporal display is for development only and should be removed before production')
        video_data = temporal.process(video_data)  # This will open the display window
        
        # Display processing summary
        if 'processing_history' in video_data.metadata:
            click.echo('\nProcessing Summary:')
            for step in video_data.metadata['processing_history']:
                click.echo(f"  - {step['component']}: {step['timestamp']}")
        
        # For now, just save some basic info about the processed video
        if output_path:
            # TODO: Add serialization/export functionality in future phases
            click.echo(f'Note: Full export functionality will be added in Phase 2')
            click.echo(f'Processed video ready: {video_data.frame_count} frames, {video_data.resolution}')
        
        click.echo('Phase 1 processing complete!')
        
        # Clean up logging resources
        LogManager.log_info('CLI', 'Processing completed successfully')
        click.echo(f'📋 Complete log available at: {LogManager.get_log_file_path()}')
        LogManager.cleanup()
        
    except Exception as e:
        # Log critical error before cleanup - LogManager is available due to import in try block
        try:
            LogManager.log_error('CLI', f'Critical processing error: {str(e)}', e)
            click.echo(f'📋 Error log available at: {LogManager.get_log_file_path()}')
            LogManager.cleanup()
        except:
            # LogManager not available, skip logging cleanup
            pass
        
        click.echo(f'Error: {str(e)}', err=True)
        click.echo(f'Traceback:\n{traceback.format_exc()}', err=True)
        raise click.Abort()


if __name__ == '__main__':
    main()