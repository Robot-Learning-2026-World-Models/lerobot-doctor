"""Tests for video integrity check."""

import json
import shutil
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot_doctor.checks.videos import check_videos
from lerobot_doctor.dataset_loader import load_local
from lerobot_doctor.runner import Severity
from tests.conftest import create_dataset


def test_no_video_features(tmp_dataset):
    """Dataset without video features should pass."""
    ds = load_local(tmp_dataset)
    result = check_videos(ds)
    assert result.severity == Severity.PASS
    assert any("No video features" in m.message for m in result.messages)


def test_video_feature_declared_but_no_path(tmp_path):
    """Video feature declared but no video_path in info.json."""
    root = create_dataset(tmp_path / "dataset")
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.images.top"] = {"dtype": "video", "shape": [3, 480, 640], "names": None}
    info_path.write_text(json.dumps(info))
    ds = load_local(root)
    result = check_videos(ds)
    assert result.severity == Severity.FAIL
    assert any("video_path" in m.message for m in result.messages)


def test_video_files_missing(tmp_path):
    """Video feature and path declared but files don't exist."""
    root = create_dataset(tmp_path / "dataset")
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.images.top"] = {"dtype": "video", "shape": [3, 480, 640], "names": None}
    info["video_path"] = "videos/{video_key}/chunk-{episode_chunk:03d}/file-{episode_index:03d}.mp4"
    info_path.write_text(json.dumps(info))
    ds = load_local(root)
    result = check_videos(ds)
    assert result.severity == Severity.FAIL
    assert any("missing" in m.message.lower() for m in result.messages)


def test_no_info(tmp_path):
    """Can't check videos without info.json."""
    root = tmp_path / "empty"
    root.mkdir()
    (root / "meta").mkdir()
    ds = load_local(root)
    result = check_videos(ds)
    assert result.severity == Severity.FAIL


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_chunked_video_frame_count_uses_sum_of_episode_windows(tmp_path):
    root = create_dataset(tmp_path / "dataset", n_episodes=2, n_frames_per_ep=10, fps=10)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.images.front"] = {
        "dtype": "video",
        "shape": [16, 16, 3],
        "names": ["height", "width", "channels"],
        "info": {"video.height": 16, "video.width": 16, "video.fps": 10},
    }
    info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    info_path.write_text(json.dumps(info))

    meta_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(meta_path)
    table = table.append_column("videos/observation.images.front/chunk_index", pa.array([0, 0]))
    table = table.append_column("videos/observation.images.front/file_index", pa.array([0, 0]))
    table = table.append_column("videos/observation.images.front/from_timestamp", pa.array([0.0, 1.0]))
    table = table.append_column("videos/observation.images.front/to_timestamp", pa.array([1.0, 2.0]))
    pq.write_table(table, meta_path)

    video_path = root / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4"
    video_path.parent.mkdir(parents=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=16x16:rate=10:duration=2",
            "-frames:v", "20",
            "-pix_fmt", "yuv420p",
            str(video_path),
        ],
        check=True,
    )

    ds = load_local(root)
    result = check_videos(ds)

    assert result.severity == Severity.PASS
    assert not any("frames, expected" in m.message for m in result.messages)
