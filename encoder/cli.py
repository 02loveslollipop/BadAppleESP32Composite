import click

 

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

@click.option('--cuda', '-c', is_flag=True, default=False, help='Enable CUDA support for video processing. Requires GPU CUDA support.')

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
def main(ctx, input, resolution, threshold, resample, dither, interactive, rle, mc, scanline, interlaced):
    """CLI for video and audio encoding for ESP32 (specifically for bad apple)."""
    if interlaced and (rle or mc):
        raise click.BadParameter('Interlaced mode cannot be used with RLE or MC compression techniques.')
    
    if scanline and interlaced:
        raise click.BadParameter('Scanline mode cannot be used with interlaced mode.')


    
    # Here you would call the actual encoding function with the provided parameters
    click.echo(f'Input file: {input}')
    click.echo(f'Resolution: {resolution}')
    click.echo(f'Threshold: {threshold}')
    click.echo(f'Resample technique: {resample}')
    click.echo(f'Dithering enabled: {dither}')
    click.echo(f'Interactive mode: {interactive}')
    click.echo(f'RLE compression: {rle}')
    click.echo(f'MC compression: {mc}')
    click.echo(f'Scanline mode: {scanline}')
    click.echo(f'Interlaced mode: {interlaced}')
    
if __name__ == '__main__':
    main()