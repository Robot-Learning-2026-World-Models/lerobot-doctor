"""Tests for idle-frame trimming."""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot_doctor.trim import trim_dataset
from tests.conftest import create_dataset


def test_trim_handles_vector_action_columns(tmp_path):
    dataset = create_dataset(tmp_path / "dataset", n_episodes=1, n_frames_per_ep=20)
    parquet_path = dataset / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(parquet_path)

    actions = np.zeros((20, 2), dtype=np.float32)
    actions[5:16, 0] = np.arange(11, dtype=np.float32)

    table = table.set_column(
        table.column_names.index("action"),
        "action",
        pa.array(actions.tolist()),
    )
    pq.write_table(table, parquet_path)

    result = trim_dataset(dataset, action_threshold=0.01, min_active_frames=10, dry_run=True)

    assert result.episodes_trimmed == 1
    assert result.frames_removed == 9
    assert result.episodes_removed == 0


def test_trim_resets_frame_timestamp_and_global_index(tmp_path):
    dataset = create_dataset(tmp_path / "dataset", n_episodes=2, n_frames_per_ep=10, fps=10)
    for parquet_path in sorted((dataset / "data" / "chunk-000").glob("*.parquet")):
        table = pq.read_table(parquet_path)
        actions = np.zeros((10, 2), dtype=np.float32)
        actions[2:, 0] = np.arange(1, 9, dtype=np.float32)
        table = table.set_column(table.column_names.index("action"), "action", pa.array(actions.tolist()))
        pq.write_table(table, parquet_path)

    result = trim_dataset(dataset, action_threshold=0.01, min_active_frames=1)

    assert result.frames_removed == 4
    first = pq.read_table(dataset / "data" / "chunk-000" / "file-000.parquet")
    second = pq.read_table(dataset / "data" / "chunk-000" / "file-001.parquet")
    assert first.column("frame_index").to_pylist() == list(range(8))
    assert first.column("timestamp").to_pylist() == [i / 10 for i in range(8)]
    assert first.column("index").to_pylist() == list(range(8))
    assert second.column("index").to_pylist() == list(range(8, 16))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_trim_rewrites_chunked_video_and_metadata(tmp_path):
    dataset = create_dataset(tmp_path / "dataset", n_episodes=2, n_frames_per_ep=10, fps=10)
    _add_chunked_video_metadata(dataset)
    _write_test_video(dataset / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4", frames=20, fps=10)

    for parquet_path in sorted((dataset / "data" / "chunk-000").glob("*.parquet")):
        table = pq.read_table(parquet_path)
        actions = np.zeros((10, 2), dtype=np.float32)
        actions[2:, 0] = np.arange(1, 9, dtype=np.float32)
        table = table.set_column(table.column_names.index("action"), "action", pa.array(actions.tolist()))
        pq.write_table(table, parquet_path)

    result = trim_dataset(dataset, action_threshold=0.01, min_active_frames=1)

    assert result.frames_removed == 4
    assert _count_video_frames(dataset / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4") == 16

    meta = pq.read_table(dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    assert meta.column("length").to_pylist() == [8, 8]
    assert meta.column("videos/observation.images.front/from_timestamp").to_pylist() == [0.0, 0.8]
    assert meta.column("videos/observation.images.front/to_timestamp").to_pylist() == [0.8, 1.6]


def test_trim_no_trim_videos_allows_missing_video_metadata(tmp_path):
    dataset = create_dataset(tmp_path / "dataset", n_episodes=1, n_frames_per_ep=10, fps=10)
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.images.front"] = {"dtype": "video", "shape": [16, 16, 3], "names": None}
    info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    info_path.write_text(json.dumps(info))

    result = trim_dataset(dataset, trim_videos=False, dry_run=True)

    assert not any("from_timestamp" in detail for detail in result.details)


def _add_chunked_video_metadata(dataset):
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.images.front"] = {
        "dtype": "video",
        "shape": [16, 16, 3],
        "names": ["height", "width", "channels"],
        "info": {"video.codec": "h264", "video.fps": 10, "video.height": 16, "video.width": 16},
    }
    info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    info_path.write_text(json.dumps(info))

    meta_path = dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(meta_path)
    table = table.append_column("videos/observation.images.front/chunk_index", pa.array([0, 0]))
    table = table.append_column("videos/observation.images.front/file_index", pa.array([0, 0]))
    table = table.append_column("videos/observation.images.front/from_timestamp", pa.array([0.0, 1.0]))
    table = table.append_column("videos/observation.images.front/to_timestamp", pa.array([1.0, 2.0]))
    pq.write_table(table, meta_path)


def _write_test_video(path, frames: int, fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=size=16x16:rate={fps}:duration={frames / fps}",
            "-frames:v", str(frames),
            "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True,
    )


def _count_video_frames(path) -> int:
    import av

    container = av.open(str(path))
    try:
        return sum(1 for _ in container.decode(video=0))
    finally:
        container.close()
