"""Dataset subsampling -- lower LeRobot capture FPS by keeping an even frame cadence."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot_doctor.trim import (
    _choose_ffmpeg_encoder,
    _format_duration,
    _format_video_path,
    _has_videotoolbox_hwaccel,
    _load_episode_meta_rows,
    _load_fps,
    _load_info,
)


@dataclass
class SubsampleResult:
    source_fps: int | None
    target_fps: int
    frames_removed: int
    details: list[str]


@dataclass
class EpisodeSubsamplePlan:
    episode_index: int
    old_length: int
    new_length: int
    rows_to_keep: list[int]


@dataclass
class TableSubsamplePlan:
    path: Path
    table: pa.Table
    rows_to_keep: list[int]


@dataclass
class VideoSubsampleSlice:
    episode_index: int
    source_path: Path
    start_ts: float
    end_ts: float
    new_length: int


def subsample_dataset(
    root: Path,
    target_fps: int,
    subsample_videos: bool = True,
    dry_run: bool = False,
    progress: Callable[[str], None] | None = None,
) -> SubsampleResult:
    """Subsample a local LeRobot dataset to a lower FPS."""
    source_fps = _load_fps(root)
    result = SubsampleResult(source_fps=source_fps, target_fps=target_fps, frames_removed=0, details=[])

    if source_fps is None:
        result.details.append("Error: info.json fps is required to subsample")
        return result
    if target_fps <= 0:
        result.details.append(f"Error: target fps must be positive, got {target_fps}")
        return result
    if target_fps > source_fps:
        result.details.append(f"Error: target fps {target_fps} exceeds source fps {source_fps}")
        return result
    if source_fps % target_fps != 0:
        result.details.append(f"Error: target fps {target_fps} must divide source fps {source_fps}")
        return result
    if target_fps == source_fps:
        result.details.append(f"Target fps equals source fps ({source_fps}); no subsampling needed")
        return result

    data_dir = root / "data"
    if not data_dir.exists():
        return result

    try:
        _emit(progress, "Scanning data parquet files for subsampling...")
        table_plans, episode_plans = _build_subsample_plan(data_dir, source_fps, target_fps, result)
        _emit(progress, f"Planned subsampling for {len(episode_plans)} episode(s)")
        video_rewrites, video_windows = (
            _build_video_plan(root, episode_plans, source_fps, target_fps) if subsample_videos else ({}, {})
        )
    except RuntimeError as e:
        result.details.append(f"Error: {e}")
        return result

    if subsample_videos and video_rewrites:
        result.details.append(f"{len(video_rewrites)} video chunk(s) will be rewritten with ffmpeg")

    if dry_run:
        return result

    if subsample_videos and video_rewrites:
        try:
            encoder = _rewrite_video_chunks(video_rewrites, source_fps, target_fps, progress=progress)
            _update_video_codec(root, encoder, target_fps)
        except RuntimeError as e:
            result.details.append(f"Error: {e}")
            return result

    _apply_table_plans(table_plans, target_fps, progress=progress)
    _emit(progress, "Updating metadata for subsampled dataset...")
    _update_metadata_after_subsample(root, episode_plans, video_windows, target_fps)
    return result


def _build_subsample_plan(
    data_dir: Path,
    source_fps: int,
    target_fps: int,
    result: SubsampleResult,
) -> tuple[list[TableSubsamplePlan], dict[int, EpisodeSubsamplePlan]]:
    table_plans: list[TableSubsamplePlan] = []
    episode_plans: dict[int, EpisodeSubsamplePlan] = {}

    for pf in sorted(data_dir.rglob("*.parquet")):
        table = pq.read_table(pf)
        if "episode_index" not in table.column_names:
            table_plans.append(TableSubsamplePlan(pf, table, list(range(len(table)))))
            continue

        ep_col = table.column("episode_index").to_pylist()
        episodes: dict[int, list[int]] = {}
        for i, ep in enumerate(ep_col):
            episodes.setdefault(int(ep), []).append(i)

        rows_to_keep: list[int] = []
        for ep_idx in sorted(episodes.keys()):
            row_indices = episodes[ep_idx]
            step = source_fps // target_fps
            kept_rows = [row for offset, row in enumerate(row_indices) if offset % step == 0]
            old_length = len(row_indices)
            new_length = len(kept_rows)
            removed = old_length - new_length
            rows_to_keep.extend(kept_rows)
            episode_plans[ep_idx] = EpisodeSubsamplePlan(ep_idx, old_length, new_length, kept_rows)
            result.frames_removed += removed
            if removed > 0:
                result.details.append(
                    f"Episode {ep_idx}: subsampled {old_length} -> {new_length} frames "
                    f"({source_fps}fps -> {target_fps}fps)"
                )

        table_plans.append(TableSubsamplePlan(pf, table, rows_to_keep))

    return table_plans, episode_plans


def _apply_table_plans(
    table_plans: list[TableSubsamplePlan],
    target_fps: int,
    progress: Callable[[str], None] | None = None,
):
    next_index = 0
    total = len(table_plans)
    for i, plan in enumerate(table_plans, start=1):
        _emit(progress, f"Writing subsampled parquet {i}/{total}: {plan.path.name}")
        table = plan.table.take(plan.rows_to_keep) if len(plan.rows_to_keep) < len(plan.table) else plan.table
        table, next_index = _reindex_table(table, target_fps, next_index)
        pq.write_table(table, plan.path)


def _reindex_table(table: pa.Table, fps: int, start_index: int) -> tuple[pa.Table, int]:
    if "episode_index" not in table.column_names:
        return table, start_index + len(table)

    ep_col = table.column("episode_index").to_pylist()
    new_frames = []
    new_timestamps = []
    current_ep = None
    counter = 0
    for ep in ep_col:
        if ep != current_ep:
            current_ep = ep
            counter = 0
        new_frames.append(counter)
        new_timestamps.append(counter / fps)
        counter += 1

    if "frame_index" in table.column_names:
        table = table.set_column(table.column_names.index("frame_index"), "frame_index", pa.array(new_frames))
    if "timestamp" in table.column_names:
        table = table.set_column(table.column_names.index("timestamp"), "timestamp", pa.array(new_timestamps))
    if "index" in table.column_names:
        indices = list(range(start_index, start_index + len(table)))
        table = table.set_column(table.column_names.index("index"), "index", pa.array(indices))

    return table, start_index + len(table)


def _build_video_plan(
    root: Path,
    episode_plans: dict[int, EpisodeSubsamplePlan],
    source_fps: int,
    target_fps: int,
) -> tuple[dict[Path, list[VideoSubsampleSlice]], dict[tuple[str, int], tuple[float, float]]]:
    info = _load_info(root)
    if not info:
        return {}, {}

    features = info.get("features", {})
    video_features = [name for name, spec in features.items() if spec.get("dtype") == "video"]
    if not video_features:
        return {}, {}
    if not info.get("video_path"):
        raise RuntimeError("video features are declared but video_path is missing in info.json")

    meta_rows = _load_episode_meta_rows(root)
    if not meta_rows:
        raise RuntimeError("video subsampling requires meta/episodes parquet metadata")

    chunks_size = info.get("chunks_size", 1000)
    rewrites: dict[Path, list[VideoSubsampleSlice]] = {}
    windows: dict[tuple[str, int], tuple[float, float]] = {}

    for feat_name in video_features:
        grouped: dict[Path, list[VideoSubsampleSlice]] = {}
        for ep_idx in sorted(episode_plans):
            plan = episode_plans[ep_idx]
            raw = meta_rows.get(ep_idx)
            if raw is None:
                raise RuntimeError(f"episode {ep_idx} is missing from meta/episodes")

            chunk_idx = raw.get(f"videos/{feat_name}/chunk_index", ep_idx // chunks_size)
            file_idx = raw.get(f"videos/{feat_name}/file_index", ep_idx)
            from_ts = raw.get(f"videos/{feat_name}/from_timestamp")
            to_ts = raw.get(f"videos/{feat_name}/to_timestamp")
            if from_ts is None or to_ts is None:
                raise RuntimeError(
                    f"episode {ep_idx} is missing videos/{feat_name}/from_timestamp or to_timestamp"
                )

            video_path = _format_video_path(info["video_path"], feat_name, int(chunk_idx), int(file_idx))
            source_path = root / video_path
            if not source_path.exists():
                raise RuntimeError(f"video file not found: {source_path}")

            grouped.setdefault(source_path, []).append(
                VideoSubsampleSlice(ep_idx, source_path, float(from_ts), float(to_ts), plan.new_length)
            )

        for source_path, slices in grouped.items():
            next_ts = 0.0
            rewrites[source_path] = slices
            for sl in slices:
                end_ts = next_ts + sl.new_length / target_fps
                windows[(feat_name, sl.episode_index)] = (next_ts, end_ts)
                next_ts = end_ts

    return rewrites, windows


def _rewrite_video_chunks(
    video_rewrites: dict[Path, list[VideoSubsampleSlice]],
    source_fps: int,
    target_fps: int,
    progress: Callable[[str], None] | None = None,
) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH; cannot subsample videos")

    encoder = _choose_ffmpeg_encoder(ffmpeg)
    use_hwaccel = _has_videotoolbox_hwaccel(ffmpeg)

    total = len(video_rewrites)
    step = source_fps // target_fps
    select_expr = f"eq(mod(n\\,{step})\\,0)"
    for chunk_i, (source_path, slices) in enumerate(video_rewrites.items(), start=1):
        if not slices:
            continue
        kept_frames = sum(sl.new_length for sl in slices)
        source_mb = source_path.stat().st_size / (1024 * 1024) if source_path.exists() else 0.0
        _emit(
            progress,
            f"Subsampling video chunk {chunk_i}/{total}: {source_path.name} "
            f"({len(slices)} episode slices, {kept_frames} frames, {source_mb:.1f} MB)",
        )
        started = time.monotonic()
        tmp_path = source_path.with_name(source_path.name + ".tmp.mp4")
        filter_parts = []
        labels = []
        for i, sl in enumerate(slices):
            label = f"v{i}"
            labels.append(f"[{label}]")
            filter_parts.append(
                f"[0:v]trim=start={sl.start_ts:.9f}:end={sl.end_ts:.9f},"
                f"setpts=PTS-STARTPTS,select='{select_expr}',"
                f"setpts=N/({target_fps}*TB)[{label}]"
            )
        filter_parts.append(f"{''.join(labels)}concat=n={len(slices)}:v=1:a=0[outv]")

        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
        if use_hwaccel:
            cmd.extend(["-hwaccel", "videotoolbox"])
        cmd.extend([
            "-i", str(source_path),
            "-filter_complex", ";".join(filter_parts),
            "-map", "[outv]",
            "-an",
            "-r", str(target_fps),
            "-c:v", encoder,
            "-pix_fmt", "yuv420p",
        ])
        if "hevc" in encoder or "x265" in encoder:
            cmd.extend(["-tag:v", "hvc1"])
        cmd.append(str(tmp_path))

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            tmp_path.replace(source_path)
            elapsed = time.monotonic() - started
            _emit(progress, f"Finished video chunk {chunk_i}/{total}: {source_path.name} in {_format_duration(elapsed)}")
        except subprocess.CalledProcessError as e:
            if tmp_path.exists():
                tmp_path.unlink()
            msg = e.stderr.strip() or e.stdout.strip() or str(e)
            raise RuntimeError(f"ffmpeg failed for {source_path.name}: {msg}") from e
    return encoder


def _update_metadata_after_subsample(
    root: Path,
    episode_plans: dict[int, EpisodeSubsamplePlan],
    video_windows: dict[tuple[str, int], tuple[float, float]],
    target_fps: int,
):
    _update_info(root, episode_plans, target_fps)
    _update_episode_metadata(root, episode_plans, video_windows)


def _update_info(root: Path, episode_plans: dict[int, EpisodeSubsamplePlan], target_fps: int):
    info = _load_info(root)
    if not info:
        return
    info["fps"] = target_fps
    info["total_frames"] = sum(p.new_length for p in episode_plans.values())
    for spec in info.get("features", {}).values():
        if spec.get("dtype") == "video":
            video_info = spec.setdefault("info", {})
            video_info["video.fps"] = target_fps
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))


def _update_video_codec(root: Path, encoder: str, target_fps: int):
    info = _load_info(root)
    if not info:
        return
    changed = False
    for spec in info.get("features", {}).values():
        if spec.get("dtype") == "video":
            video_info = spec.setdefault("info", {})
            video_info["video.codec"] = "hevc" if ("hevc" in encoder or "x265" in encoder) else "h264"
            video_info["video.pix_fmt"] = "yuv420p"
            video_info["video.fps"] = target_fps
            changed = True
    if changed:
        (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))


def _update_episode_metadata(
    root: Path,
    episode_plans: dict[int, EpisodeSubsamplePlan],
    video_windows: dict[tuple[str, int], tuple[float, float]],
):
    meta_dir = root / "meta" / "episodes"
    if not meta_dir.exists():
        return

    global_from: dict[int, int] = {}
    cursor = 0
    for ep_idx in sorted(episode_plans):
        plan = episode_plans[ep_idx]
        global_from[ep_idx] = cursor
        cursor += plan.new_length

    for pf in sorted(meta_dir.rglob("*.parquet")):
        table = pq.read_table(pf)
        if "episode_index" not in table.column_names:
            continue
        ep_col = [int(ep) for ep in table.column("episode_index").to_pylist()]
        updates: dict[str, list] = {}
        if "length" in table.column_names:
            updates["length"] = [
                episode_plans[ep].new_length if ep in episode_plans else table.column("length")[i].as_py()
                for i, ep in enumerate(ep_col)
            ]
        if "dataset_from_index" in table.column_names:
            updates["dataset_from_index"] = [
                global_from.get(ep, table.column("dataset_from_index")[i].as_py())
                for i, ep in enumerate(ep_col)
            ]
        if "dataset_to_index" in table.column_names:
            updates["dataset_to_index"] = [
                global_from[ep] + episode_plans[ep].new_length
                if ep in episode_plans else table.column("dataset_to_index")[i].as_py()
                for i, ep in enumerate(ep_col)
            ]

        for col in table.column_names:
            if not col.startswith("videos/") or not col.endswith(("/from_timestamp", "/to_timestamp")):
                continue
            feature = col.removeprefix("videos/").rsplit("/", 1)[0]
            is_from = col.endswith("/from_timestamp")
            values = []
            for i, ep in enumerate(ep_col):
                window = video_windows.get((feature, ep))
                values.append((window[0] if is_from else window[1]) if window else table.column(col)[i].as_py())
            updates[col] = values

        for col, values in updates.items():
            table = table.set_column(table.column_names.index(col), col, pa.array(values))
        pq.write_table(table, pf)


def _emit(progress: Callable[[str], None] | None, message: str):
    if progress is not None:
        progress(message)
