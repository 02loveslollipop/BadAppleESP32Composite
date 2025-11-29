import sys
import types

# Provide a lightweight stub for the `av` module so imports of VideoOpener/chainable
# succeed even when PyAV isn't installed in the test environment.
if "av" not in sys.modules:
    fake_av = types.SimpleNamespace()

    def _unconfigured_open(*args, **kwargs):
        raise RuntimeError("Fake av stub: configure av.open in the test before use")

    fake_av.open = _unconfigured_open
    sys.modules["av"] = fake_av

import numpy as np
import pytest

from encoder.chainable.basex import VideoData


@pytest.fixture
def sample_rgb_video() -> VideoData:
    """Small RGB clip for resize/tests."""
    frames = np.arange(3 * 2 * 4 * 3, dtype=np.uint8).reshape(3, 2, 4, 3)
    return VideoData(
        frames=frames,
        frame_rate=30.0,
        resolution=(4, 2),
        color_mode="RGB",
        metadata={},
    )


@pytest.fixture
def sample_gray_video() -> VideoData:
    """Small grayscale clip for reframe and 1-bit conversion tests."""
    frames = np.arange(9 * 2 * 2 * 1, dtype=np.uint8).reshape(9, 2, 2, 1)
    return VideoData(
        frames=frames,
        frame_rate=30.0,
        resolution=(2, 2),
        color_mode="GRAY",
        metadata={},
    )
