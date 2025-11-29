import importlib
import sys
import types

import numpy as np


def test_video_opener_uses_stubbed_av(monkeypatch, tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"stub")

    class FakeStream:
        width = 4
        height = 3
        average_rate = 30
        duration = 90
        time_base = 1 / 30
        frames = 90
        codec = types.SimpleNamespace(name="h264")
        pix_fmt = "yuv420p"

    class FakeContainer:
        def __init__(self):
            self.streams = types.SimpleNamespace(video=[FakeStream()])
            self.closed = False

        def close(self):
            self.closed = True

    # Replace av.open on the shared stub with our fake container
    sys.modules["av"].open = lambda path: FakeContainer()

    import encoder.chainable.openx as openx

    # Reload to make sure the stubbed av module is used
    openx = importlib.reload(openx)

    fake_frames = np.ones((2, FakeStream.height, FakeStream.width, 3), dtype=np.uint8)
    monkeypatch.setattr(
        openx.VideoOpener,
        "_extract_frames",
        lambda self, container, stream, est: fake_frames,
    )

    opener = openx.VideoOpener(video_path, color_mode="RGB")
    data = opener.process()

    assert data.frames.shape == fake_frames.shape
    assert data.resolution == (FakeStream.width, FakeStream.height)
    assert data.frame_rate == float(FakeStream.average_rate)
    assert data.metadata["source_file"] == str(video_path)
    assert data.metadata["codec"] == FakeStream.codec.name
    assert data.metadata["pixel_format"] == FakeStream.pix_fmt
    assert data.metadata["processing_history"][0]["component"] == "VideoOpener"
