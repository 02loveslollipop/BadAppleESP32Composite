import numpy as np

from encoder.chainable.basex import VideoData
from encoder.chainable.convert1bitx import Video1BitConverter


def test_video_1bit_converter_thresholding():
    frames = np.array([[[0, 128], [200, 255]]], dtype=np.uint8)
    data = VideoData(
        frames=frames,
        frame_rate=24.0,
        resolution=(2, 2),
        color_mode="GRAY",
        metadata={},
    )

    converter = Video1BitConverter(threshold=128, dither_method="none", use_gpu=False)
    result = converter.process(data)

    expected = np.array([[[0, 0], [1, 1]]], dtype=np.uint8)
    assert np.array_equal(result.frames, expected)
    assert result.color_mode == "BW"
    assert result.metadata["processing_history"][-1]["component"] == "Video1BitConverter"


def test_video_1bit_converter_ordered_dither_pattern():
    frames = np.full((1, 2, 2), 120, dtype=np.uint8)
    data = VideoData(
        frames=frames,
        frame_rate=24.0,
        resolution=(2, 2),
        color_mode="GRAY",
        metadata={},
    )

    converter = Video1BitConverter(
        threshold=128,
        dither_method="ordered",
        dither_pattern="bayer2x2",
        use_gpu=False,
    )
    result = converter.process(data)

    expected = np.array([[[1, 0], [0, 1]]], dtype=np.uint8)
    assert np.array_equal(result.frames, expected)
    assert result.color_mode == "BW"
