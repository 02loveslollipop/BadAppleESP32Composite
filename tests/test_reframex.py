import numpy as np
import pytest

from encoder.chainable.reframex import ProcessingError, VideoReframer


def test_video_reframer_simple_downsample(sample_gray_video, monkeypatch):
    def fake_resample(frames, source_fps, target_fps, technique):
        # Drop every third frame to simulate downsampling
        indices = np.arange(0, frames.shape[0], 3, dtype=np.int32)
        stats = {
            "original_frame_count": frames.shape[0],
            "output_frame_count": len(indices),
            "compression_ratio": len(indices) / frames.shape[0],
            "source_fps": source_fps,
            "target_fps": target_fps,
            "technique": technique,
            "frame_drop_percentage": (1 - len(indices) / frames.shape[0]) * 100,
        }
        return frames[indices], indices, stats

    reframer = VideoReframer(target_fps=10, technique="simple", use_gpu=False)
    monkeypatch.setattr(reframer, "resample_function", fake_resample)

    result = reframer.process(sample_gray_video)

    assert result.frame_rate == 10
    assert result.frames.shape[0] == 3
    assert np.array_equal(result.frames, sample_gray_video.frames[[0, 3, 6]])
    assert result.metadata["reframe_statistics"]["output_frame_count"] == 3


def test_video_reframer_rejects_higher_target(sample_gray_video):
    reframer = VideoReframer(target_fps=60, technique="simple", use_gpu=False)

    with pytest.raises(ProcessingError):
        reframer.process(sample_gray_video)
