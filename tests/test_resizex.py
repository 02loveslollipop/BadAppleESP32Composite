from encoder.chainable.resizex import VideoResizer


def test_video_resizer_changes_resolution_and_tracks_metadata(sample_rgb_video):
    resizer = VideoResizer(
        target_resolution=(2, 1),
        lanczos_kernel=4,
        use_gpu=False,
        workers=1,
        batch_size=2,
    )

    resized = resizer.process(sample_rgb_video)

    assert resized.frames.shape == (sample_rgb_video.frame_count, 1, 2, 3)
    assert resized.resolution == (2, 1)
    assert resized.color_mode == sample_rgb_video.color_mode
    history = resized.metadata.get("processing_history", [])
    assert any(step["component"] == "VideoResizer" for step in history)


def test_video_resizer_noop_when_resolution_matches(sample_rgb_video):
    resizer = VideoResizer(
        target_resolution=sample_rgb_video.resolution,
        lanczos_kernel=4,
        use_gpu=False,
        workers=1,
        batch_size=2,
    )

    result = resizer.process(sample_rgb_video)
    assert result is sample_rgb_video
