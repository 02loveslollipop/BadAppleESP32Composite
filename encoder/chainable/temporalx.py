"""
Video temporal visualization component using matplotlib for real-time display.

This component provides real-time video playback and visualization capabilities,
allowing users to preview processed video data with configurable playback options.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional, Dict, Any, Tuple, Callable
import time
import threading
from collections import deque

from . import ChainComponent, VideoData, ProcessingError, LogManager


class VideoTemporal(ChainComponent):
    """Component for real-time video visualization and playback."""
    
    def __init__(self,
                 playback_fps: Optional[float] = None,
                 window_title: str = "Video Temporal Display",
                 figure_size: Tuple[int, int] = (10, 8),
                 show_info: bool = True,
                 show_controls: bool = True,
                 auto_play: bool = True,
                 loop: bool = True,
                 colormap: str = 'gray',
                 interpolation: str = 'nearest'):
        """
        Initialize the video temporal display.
        
        Args:
            playback_fps: Override frame rate for playback (uses video FPS if None)
            window_title: Title for the display window
            figure_size: Size of the matplotlib figure (width, height)
            show_info: Whether to show frame info overlay
            show_controls: Whether to show playback controls
            auto_play: Whether to start playing automatically
            loop: Whether to loop the video playback
            colormap: Matplotlib colormap for grayscale videos
            interpolation: Image interpolation method
        """
        super().__init__("VideoTemporal")
        
        self.playback_fps = playback_fps
        self.window_title = window_title
        self.figure_size = figure_size
        self.show_info = show_info
        self.show_controls = show_controls
        self.auto_play = auto_play
        self.loop = loop
        self.colormap = colormap
        self.interpolation = interpolation
        
        # Playback state
        self.is_playing = auto_play
        self.current_frame = 0
        self.frames = None
        self.video_data = None
        self.animation = None
        self.fig = None
        self.ax = None
        self.img_display = None
        self.info_text = None
        
        # Performance tracking
        self.frame_times = deque(maxlen=30)  # Track last 30 frame times
        self.last_frame_time = time.time()
        
        LogManager.log_info(self.name, f"Initialized: playback_fps={playback_fps}, auto_play={auto_play}")
    
    def _validate_input(self, data: VideoData):
        """Validate input data for temporal display."""
        super()._validate_input(data)
        
        if data.frames.size == 0:
            raise ProcessingError(
                "No frames to display",
                component=self.name
            )
        
        if len(data.frames.shape) not in [3, 4]:
            raise ProcessingError(
                f"Invalid frame data shape: {data.frames.shape}. Expected (N, H, W) or (N, H, W, C)",
                component=self.name
            )
    
    def _setup_display(self, data: VideoData):
        """Setup the matplotlib display components."""
        # Close any existing figure
        if self.fig is not None:
            plt.close(self.fig)
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=self.figure_size)
        self.fig.canvas.manager.set_window_title(self.window_title)
        
        # Prepare first frame for display
        first_frame = self._prepare_frame_for_display(data.frames[0], data.color_mode)
        
        # Create image display
        if data.color_mode == 'RGB':
            self.img_display = self.ax.imshow(first_frame, interpolation=self.interpolation)
        else:
            self.img_display = self.ax.imshow(first_frame, cmap=self.colormap, interpolation=self.interpolation)
        
        # Configure axes
        self.ax.set_title(f"Frame 1/{data.frame_count}")
        self.ax.axis('off')  # Hide axes for cleaner display
        
        # Add info text if requested
        if self.show_info:
            info_str = self._get_frame_info(0, data)
            self.info_text = self.fig.text(0.02, 0.98, info_str, 
                                         fontsize=10, verticalalignment='top',
                                         bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
                                         color='white', family='monospace')
        
        # Add controls if requested
        if self.show_controls:
            self._setup_controls()
        
        # Tight layout
        plt.tight_layout()
        
        LogManager.log_info(self.name, f"Display setup complete: {data.resolution}, {data.frame_count} frames")
    
    def _prepare_frame_for_display(self, frame: np.ndarray, color_mode: str) -> np.ndarray:
        """Prepare a frame for matplotlib display."""
        if color_mode == 'RGB':
            # Ensure RGB frame is in correct format
            if frame.ndim == 3 and frame.shape[2] == 3:
                # Normalize to 0-1 if needed
                if frame.dtype == np.uint8:
                    return frame.astype(np.float32) / 255.0
                return frame
            else:
                raise ProcessingError(f"Invalid RGB frame shape: {frame.shape}", component=self.name)
        
        elif color_mode in ['GRAY', 'BW']:
            # Handle grayscale/binary frames
            if frame.ndim == 2:
                return frame
            elif frame.ndim == 3 and frame.shape[2] == 1:
                return frame.squeeze(axis=2)
            else:
                raise ProcessingError(f"Invalid grayscale frame shape: {frame.shape}", component=self.name)
        
        else:
            raise ProcessingError(f"Unsupported color mode: {color_mode}", component=self.name)
    
    def _get_frame_info(self, frame_idx: int, data: VideoData) -> str:
        """Generate frame information string."""
        current_time = frame_idx / data.frame_rate if data.frame_rate > 0 else 0
        
        # Calculate current FPS if we have frame timing data
        current_fps = "N/A"
        if len(self.frame_times) > 1:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            current_fps = f"{1.0 / avg_frame_time:.1f}"
        
        info_lines = [
            f"Frame: {frame_idx + 1}/{data.frame_count}",
            f"Time: {current_time:.2f}s",
            f"Resolution: {data.resolution[0]}x{data.resolution[1]}",
            f"Video FPS: {data.frame_rate:.1f}",
            f"Display FPS: {current_fps}",
            f"Color: {data.color_mode}",
        ]
        
        # Add processing history if available
        if 'processing_history' in data.metadata and data.metadata['processing_history']:
            last_process = data.metadata['processing_history'][-1]['component']
            info_lines.append(f"Last Process: {last_process}")
        
        return "\n".join(info_lines)
    
    def _setup_controls(self):
        """Setup playback control buttons."""
        # Add control buttons at the bottom
        ax_play = plt.axes([0.2, 0.02, 0.1, 0.04])
        ax_pause = plt.axes([0.31, 0.02, 0.1, 0.04])
        ax_reset = plt.axes([0.42, 0.02, 0.1, 0.04])
        ax_step = plt.axes([0.53, 0.02, 0.1, 0.04])
        
        from matplotlib.widgets import Button
        
        self.btn_play = Button(ax_play, 'Play')
        self.btn_pause = Button(ax_pause, 'Pause')
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_step = Button(ax_step, 'Step')
        
        # Connect button events
        self.btn_play.on_clicked(lambda x: self._control_play())
        self.btn_pause.on_clicked(lambda x: self._control_pause())
        self.btn_reset.on_clicked(lambda x: self._control_reset())
        self.btn_step.on_clicked(lambda x: self._control_step())
        
        # Keyboard controls
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
    
    def _control_play(self):
        """Start playback."""
        self.is_playing = True
        LogManager.log_info(self.name, "Playback started")
    
    def _control_pause(self):
        """Pause playback."""
        self.is_playing = False
        LogManager.log_info(self.name, "Playback paused")
    
    def _control_reset(self):
        """Reset to first frame."""
        self.current_frame = 0
        self.is_playing = self.auto_play
        LogManager.log_info(self.name, "Playback reset")
    
    def _control_step(self):
        """Step to next frame."""
        if self.frames is not None:
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            self._update_frame(self.current_frame)
            LogManager.log_info(self.name, f"Stepped to frame {self.current_frame + 1}")
    
    def _on_key_press(self, event):
        """Handle keyboard events."""
        if event.key == ' ':  # Spacebar to toggle play/pause
            if self.is_playing:
                self._control_pause()
            else:
                self._control_play()
        elif event.key == 'r':  # R to reset
            self._control_reset()
        elif event.key == 'right':  # Right arrow to step forward
            self._control_step()
        elif event.key == 'left':  # Left arrow to step backward
            if self.frames is not None:
                self.current_frame = (self.current_frame - 1) % len(self.frames)
                self._update_frame(self.current_frame)
    
    def _update_frame(self, frame_idx: int):
        """Update the displayed frame."""
        if self.frames is None or self.img_display is None:
            return
        
        # Prepare frame for display
        frame = self._prepare_frame_for_display(self.frames[frame_idx], self.video_data.color_mode)
        
        # Update image
        self.img_display.set_array(frame)
        
        # Update title
        self.ax.set_title(f"Frame {frame_idx + 1}/{len(self.frames)}")
        
        # Update info text
        if self.show_info and self.info_text is not None:
            info_str = self._get_frame_info(frame_idx, self.video_data)
            self.info_text.set_text(info_str)
        
        # Track frame timing
        current_time = time.time()
        if hasattr(self, 'last_frame_time'):
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
        self.last_frame_time = current_time
        
        # Redraw
        self.fig.canvas.draw_idle()
    
    def _animate_frame(self, frame_idx: int):
        """Animation function for matplotlib FuncAnimation."""
        if not self.is_playing:
            return [self.img_display]
        
        # Update current frame
        self.current_frame = frame_idx % len(self.frames)
        self._update_frame(self.current_frame)
        
        # Handle looping
        if not self.loop and frame_idx >= len(self.frames) - 1:
            self.is_playing = False
            LogManager.log_info(self.name, "Playback completed (no loop)")
        
        return [self.img_display]
    
    def process(self, data: VideoData) -> VideoData:
        """
        Display video frames with temporal visualization.
        
        Args:
            data: Input VideoData
            
        Returns:
            VideoData (unchanged - this is a display component)
        """
        self._validate_input(data)
        
        # Store video data and frames
        self.video_data = data
        self.frames = data.frames
        self.current_frame = 0
        
        # Determine playback FPS
        effective_fps = self.playback_fps if self.playback_fps is not None else data.frame_rate
        frame_interval = 1000 / effective_fps if effective_fps > 0 else 50  # milliseconds
        
        LogManager.log_info(
            self.name, 
            f"Starting temporal display: {data.frame_count} frames at {effective_fps:.1f} FPS"
        )
        
        try:
            # Setup display
            self._setup_display(data)
            
            # Create animation
            frame_count = len(self.frames)
            frames_to_animate = range(frame_count * 10 if self.loop else frame_count)
            
            self.animation = animation.FuncAnimation(
                self.fig,
                self._animate_frame,
                frames=frames_to_animate,
                interval=frame_interval,
                blit=False,
                repeat=self.loop
            )
            
            # Show the plot
            self.logger.info(f"Displaying {frame_count} frames at {effective_fps:.1f} FPS")
            self.logger.info("Controls: SPACE=play/pause, R=reset, LEFT/RIGHT=step, ESC=close")
            
            plt.show()
            
            LogManager.log_info(self.name, "Temporal display completed")
            
            # Return original data unchanged (this is a display component)
            return data
            
        except Exception as e:
            LogManager.log_error(self.name, f"Temporal display failed: {str(e)}", e)
            raise ProcessingError(
                f"Temporal display failed: {str(e)}",
                component=self.name,
                details={'frame_count': data.frame_count, 'fps': effective_fps}
            )
    
    def __del__(self):
        """Cleanup on destruction."""
        if self.fig is not None:
            plt.close(self.fig)


# Convenience function for quick video temporal display
def display_video_temporal(data: VideoData,
                          playback_fps: Optional[float] = None,
                          window_title: str = "Video Temporal Display",
                          auto_play: bool = True,
                          **kwargs) -> VideoData:
    """
    Convenience function for video temporal display.
    
    Args:
        data: Input video data
        playback_fps: Override frame rate for playback
        window_title: Title for the display window
        auto_play: Whether to start playing automatically
        **kwargs: Additional parameters for VideoTemporal
        
    Returns:
        VideoData (unchanged)
    """
    temporal = VideoTemporal(
        playback_fps=playback_fps,
        window_title=window_title,
        auto_play=auto_play,
        **kwargs
    )
    return temporal.process(data)


__all__ = ['VideoTemporal', 'display_video_temporal']
