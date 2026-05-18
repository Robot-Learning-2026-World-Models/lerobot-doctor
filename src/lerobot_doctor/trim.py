"""Episode trimming -- remove idle/static frames from episodes."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class TrimResult:
    episodes_trimmed: int
    frames_removed: int
    episodes_removed: int
    details: list[str]


@dataclass
class EpisodeTrimShard:
    """Per-(episode, data_file) bookkeeping for trimming.

    LeRobot v2.1 episodes can span multiple data files; each shard carries its own
    portion of the episode and its own video window. The trim cut [plan.start,
    plan.end) is in episode-global coordinates and is intersected with each shard's
    range to derive per-shard kept rows, video sub-windows, and metadata updates.
    """
    data_file_index: int
    shard_start_in_episode: int   # global episode row offset of this shard's first row
    old_length: int               # rows of this episode in this data file
    kept_local_start: int = 0     # within shard, first kept row (set after trim plan known)
    kept_local_end: int = 0       # within shard, end (exclusive)
    new_length: int = 0           # kept_local_end - kept_local_start
    global_from_index: int = 0    # post-trim global row index for shard start


@dataclass
class EpisodeTrimPlan:
    episode_index: int
    parquet_path: Path
    old_length: int
    start: int
    end: int
    new_length: int
    removed: int
    source_start_ts: float
    source_end_ts: float
    dropped: bool = False
    shards: list[EpisodeTrimShard] = field(default_factory=list)


@dataclass
class TableTrimPlan:
    path: Path
    table: pa.Table
    rows_to_keep: list[int]


@dataclass
class VideoSlice:
    """One per-shard trim slice. start_ts/end_ts are source-video timestamps."""
    episode_index: int
    data_file_index: int
    source_path: Path
    start_ts: float
    end_ts: float
    new_length: int


def trim_dataset(
    root: Path,
    action_threshold: float = 0.01,
    min_active_frames: int = 10,
    trim_start: bool = True,
    trim_end: bool = True,
    remove_fully_static: bool = True,
    trim_videos: bool = True,
    dry_run: bool = False,
    workers: int = 4,
    progress: Callable[[str], None] | None = None,
) -> TrimResult:
    """Trim idle frames from episodes."""
    result = TrimResult(episodes_trimmed=0, frames_removed=0, episodes_removed=0, details=[])

    data_dir = root / "data"
    if not data_dir.exists():
        return result

    fps = _load_fps(root)
    try:
        _emit(progress, "Scanning data parquet files...")
        table_plans, episode_plans = _build_trim_plan(
            data_dir,
            action_threshold=action_threshold,
            min_active_frames=min_active_frames,
            trim_start=trim_start,
            trim_end=trim_end,
            remove_fully_static=remove_fully_static,
            fps=fps,
            result=result,
        )
        _emit(progress, f"Planned trims for {len(episode_plans)} episode(s)")
        video_rewrites, video_windows = _build_video_plan(root, episode_plans) if trim_videos else ({}, {})
    except RuntimeError as e:
        result.details.append(f"Error: {e}")
        return result

    if trim_videos and video_rewrites:
        result.details.append(f"{len(video_rewrites)} video chunk(s) will be rewritten with ffmpeg")

    if dry_run:
        return result

    if fps is None:
        result.details.append("Error: info.json fps is required to reset timestamps")
        return result

    if trim_videos and video_rewrites:
        try:
            encoder = _rewrite_video_chunks(video_rewrites, fps, workers=workers, progress=progress)
            _update_video_codec(root, encoder)
        except RuntimeError as e:
            result.details.append(f"Error: {e}")
            return result

    _apply_table_plans(table_plans, fps, progress=progress)
    if result.frames_removed > 0 or result.episodes_removed > 0:
        _emit(progress, "Updating metadata...")
        _update_metadata_after_trim(root, episode_plans, video_windows)

    return result


def _build_trim_plan(
    data_dir: Path,
    action_threshold: float,
    min_active_frames: int,
    trim_start: bool,
    trim_end: bool,
    remove_fully_static: bool,
    fps: int | None,
    result: TrimResult,
) -> tuple[list[TableTrimPlan], dict[int, EpisodeTrimPlan]]:
    tables: list[tuple[Path, pa.Table]] = []
    episode_entries: dict[int, list[tuple[int, int]]] = {}
    table_rows_to_keep: list[list[int]] = []

    for pf in sorted(data_dir.rglob("*.parquet")):
        table = pq.read_table(pf)
        table_idx = len(tables)
        tables.append((pf, table))
        table_rows_to_keep.append([])

        if "episode_index" not in table.column_names:
            table_rows_to_keep[table_idx].extend(range(len(table)))
            continue

        ep_col = table.column("episode_index").to_pylist()
        for row_idx, ep in enumerate(ep_col):
            episode_entries.setdefault(int(ep), []).append((table_idx, row_idx))

    episode_plans: dict[int, EpisodeTrimPlan] = {}
    # Track ordered list of (parquet_idx, ep_idx) for global_from_index assignment.
    parquet_episode_order: list[tuple[int, int]] = []
    seen_in_parquet: dict[int, set[int]] = {}
    for table_idx, (pf, table) in enumerate(tables):
        if "episode_index" not in table.column_names:
            continue
        ep_seq = table.column("episode_index").to_pylist()
        seen = seen_in_parquet.setdefault(table_idx, set())
        for ep in ep_seq:
            iep = int(ep)
            if iep not in seen:
                seen.add(iep)
                parquet_episode_order.append((table_idx, iep))

    for ep_idx in sorted(episode_entries):
        entries = episode_entries[ep_idx]
        first_path = tables[entries[0][0]][0]
        action_cols = _action_cols_for_entries(tables, entries)
        if not action_cols:
            plan = EpisodeTrimPlan(
                ep_idx,
                first_path,
                len(entries),
                0,
                len(entries),
                len(entries),
                0,
                *_source_time_bounds_entries(tables, entries, 0, len(entries), fps),
            )
        else:
            plan = _plan_episode_entries(
                tables,
                first_path,
                ep_idx,
                entries,
                action_cols,
                action_threshold,
                min_active_frames,
                trim_start,
                trim_end,
                remove_fully_static,
                fps,
            )
        episode_plans[ep_idx] = plan

        if plan.dropped:
            result.episodes_removed += 1
            result.frames_removed += plan.old_length
            result.details.append(f"Episode {ep_idx}: removed (fully static)")
            continue

        for table_idx, row_idx in entries[plan.start:plan.end]:
            table_rows_to_keep[table_idx].append(row_idx)
        if plan.removed > 0:
            result.episodes_trimmed += 1
            result.frames_removed += plan.removed
            result.details.append(
                f"Episode {ep_idx}: trimmed {plan.removed} idle frames "
                f"({plan.old_length} -> {plan.new_length})"
            )

    _attach_shards(tables, episode_entries, episode_plans, parquet_episode_order)

    table_plans = [
        TableTrimPlan(path, table, sorted(rows_to_keep))
        for (path, table), rows_to_keep in zip(tables, table_rows_to_keep)
    ]
    return table_plans, episode_plans


def _attach_shards(
    tables: list[tuple[Path, pa.Table]],
    episode_entries: dict[int, list[tuple[int, int]]],
    episode_plans: dict[int, EpisodeTrimPlan],
    parquet_episode_order: list[tuple[int, int]],
) -> None:
    """Compute per-shard intersections with the trim range and global_from_index."""
    # Map table_idx -> data_file_index, parsed from the parquet path once.
    table_data_file: dict[int, int] = {}
    for table_idx, (pf, _) in enumerate(tables):
        try:
            _, file_idx = _parse_data_path(pf)
        except RuntimeError:
            continue
        table_data_file[table_idx] = file_idx

    # Per episode, walk entries and group consecutive same-table_idx runs as shards.
    # `entries` from episode_entries is already in (parquet_sort, original-row) order
    # because _build_trim_plan iterates parquets in sorted order.
    shards_by_episode: dict[int, list[EpisodeTrimShard]] = {}
    for ep_idx, entries in episode_entries.items():
        plan = episode_plans.get(ep_idx)
        if plan is None:
            continue
        shards: list[EpisodeTrimShard] = []
        cursor = 0
        for table_idx, group in _group_consecutive(entries):
            file_idx = table_data_file.get(table_idx)
            if file_idx is None:
                raise RuntimeError(
                    f"episode {ep_idx} references parquet {tables[table_idx][0]} that does not match "
                    "the expected chunk-XXX/file-YYY layout"
                )
            shard = EpisodeTrimShard(
                data_file_index=file_idx,
                shard_start_in_episode=cursor,
                old_length=len(group),
            )
            # Intersect [plan.start, plan.end) with [cursor, cursor + len(group)) in episode coords.
            if not plan.dropped:
                global_start = max(plan.start, cursor)
                global_end = min(plan.end, cursor + len(group))
                if global_end > global_start:
                    shard.kept_local_start = global_start - cursor
                    shard.kept_local_end = global_end - cursor
                    shard.new_length = shard.kept_local_end - shard.kept_local_start
            shards.append(shard)
            cursor += len(group)
        shards_by_episode[ep_idx] = shards
        plan.shards = shards

    # Assign global_from_index in (parquet_sort, episode_sort within parquet) order so
    # it matches the order _apply_table_plans writes rows post-trim.
    cumulative = 0
    for table_idx, ep_idx in parquet_episode_order:
        plan = episode_plans.get(ep_idx)
        if plan is None or plan.dropped:
            continue
        file_idx = table_data_file.get(table_idx)
        if file_idx is None:
            continue
        for shard in plan.shards:
            if shard.data_file_index != file_idx:
                continue
            shard.global_from_index = cumulative
            cumulative += shard.new_length


def _group_consecutive(entries: list[tuple[int, int]]):
    if not entries:
        return
    current_idx = entries[0][0]
    bucket: list[tuple[int, int]] = []
    for e in entries:
        if e[0] != current_idx:
            yield current_idx, bucket
            current_idx = e[0]
            bucket = []
        bucket.append(e)
    yield current_idx, bucket


def _action_cols_for_entries(
    tables: list[tuple[Path, pa.Table]],
    entries: list[tuple[int, int]],
) -> list[str]:
    for table_idx, _ in entries:
        table = tables[table_idx][1]
        cols = [c for c in table.column_names if c.startswith("action")]
        if cols:
            return cols
    return []


def _plan_episode_entries(
    tables: list[tuple[Path, pa.Table]],
    parquet_path: Path,
    ep_idx: int,
    entries: list[tuple[int, int]],
    action_cols: list[str],
    action_threshold: float,
    min_active_frames: int,
    trim_start: bool,
    trim_end: bool,
    remove_fully_static: bool,
    fps: int | None,
) -> EpisodeTrimPlan:
    old_length = len(entries)
    action_matrix = _action_matrix_entries(tables, entries, action_cols)
    if action_matrix is None or len(action_matrix) < 2:
        source_start_ts, source_end_ts = _source_time_bounds_entries(tables, entries, 0, old_length, fps)
        return EpisodeTrimPlan(
            ep_idx, parquet_path, old_length, 0, old_length, old_length, 0, source_start_ts, source_end_ts
        )

    diffs = np.abs(np.diff(action_matrix, axis=0))
    activity = np.concatenate([[0], diffs.mean(axis=1)])
    active_indices = np.where(activity > action_threshold)[0]

    if len(active_indices) < min_active_frames:
        if remove_fully_static:
            return EpisodeTrimPlan(ep_idx, parquet_path, old_length, 0, 0, 0, old_length, 0.0, 0.0, dropped=True)
        source_start_ts, source_end_ts = _source_time_bounds_entries(tables, entries, 0, old_length, fps)
        return EpisodeTrimPlan(
            ep_idx, parquet_path, old_length, 0, old_length, old_length, 0, source_start_ts, source_end_ts
        )

    start = int(active_indices[0]) if trim_start else 0
    end = int(active_indices[-1]) + 1 if trim_end else old_length
    new_length = end - start
    source_start_ts, source_end_ts = _source_time_bounds_entries(tables, entries, start, end, fps)
    return EpisodeTrimPlan(
        ep_idx,
        parquet_path,
        old_length,
        start,
        end,
        new_length,
        old_length - new_length,
        source_start_ts,
        source_end_ts,
    )


def _source_time_bounds_entries(
    tables: list[tuple[Path, pa.Table]],
    entries: list[tuple[int, int]],
    start: int,
    end: int,
    fps: int | None,
) -> tuple[float, float]:
    if end <= start:
        return 0.0, 0.0
    start_table = tables[entries[start][0]][1]
    end_table = tables[entries[end - 1][0]][1]
    start_row = entries[start][1]
    end_row = entries[end - 1][1]
    if "timestamp" in start_table.column_names and "timestamp" in end_table.column_names:
        start_ts = float(start_table.column("timestamp")[start_row].as_py())
        last_ts = float(end_table.column("timestamp")[end_row].as_py())
        frame_dt = (1.0 / fps) if fps else 0.0
        return start_ts, last_ts + frame_dt
    if fps:
        return start / fps, end / fps
    return float(start), float(end)


def _action_matrix_entries(
    tables: list[tuple[Path, pa.Table]],
    entries: list[tuple[int, int]],
    action_cols: list[str],
) -> np.ndarray | None:
    actions = []
    for col in action_cols:
        try:
            vals = np.array([
                tables[table_idx][1].column(col)[row_idx].as_py()
                for table_idx, row_idx in entries
            ], dtype=np.float64)
        except (KeyError, ValueError, TypeError):
            continue
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        elif vals.ndim > 1:
            vals = vals.reshape(len(entries), -1)
        else:
            continue
        actions.append(vals)
    if not actions:
        return None
    return np.concatenate(actions, axis=1)


def _plan_episode(
    table: pa.Table,
    parquet_path: Path,
    ep_idx: int,
    row_indices: list[int],
    action_cols: list[str],
    action_threshold: float,
    min_active_frames: int,
    trim_start: bool,
    trim_end: bool,
    remove_fully_static: bool,
    fps: int | None,
) -> EpisodeTrimPlan:
    old_length = len(row_indices)
    action_matrix = _action_matrix(table, row_indices, action_cols)
    if action_matrix is None or len(action_matrix) < 2:
        source_start_ts, source_end_ts = _source_time_bounds(table, row_indices, 0, old_length, fps)
        return EpisodeTrimPlan(
            ep_idx, parquet_path, old_length, 0, old_length, old_length, 0, source_start_ts, source_end_ts
        )

    diffs = np.abs(np.diff(action_matrix, axis=0))
    activity = np.concatenate([[0], diffs.mean(axis=1)])
    active_indices = np.where(activity > action_threshold)[0]

    if len(active_indices) < min_active_frames:
        if remove_fully_static:
            return EpisodeTrimPlan(ep_idx, parquet_path, old_length, 0, 0, 0, old_length, 0.0, 0.0, dropped=True)
        source_start_ts, source_end_ts = _source_time_bounds(table, row_indices, 0, old_length, fps)
        return EpisodeTrimPlan(
            ep_idx, parquet_path, old_length, 0, old_length, old_length, 0, source_start_ts, source_end_ts
        )

    start = int(active_indices[0]) if trim_start else 0
    end = int(active_indices[-1]) + 1 if trim_end else old_length
    new_length = end - start
    source_start_ts, source_end_ts = _source_time_bounds(table, row_indices, start, end, fps)
    return EpisodeTrimPlan(
        ep_idx,
        parquet_path,
        old_length,
        start,
        end,
        new_length,
        old_length - new_length,
        source_start_ts,
        source_end_ts,
    )


def _source_time_bounds(
    table: pa.Table,
    row_indices: list[int],
    start: int,
    end: int,
    fps: int | None,
) -> tuple[float, float]:
    if end <= start:
        return 0.0, 0.0
    if "timestamp" in table.column_names:
        timestamps = table.column("timestamp")
        start_ts = float(timestamps[row_indices[start]].as_py())
        last_ts = float(timestamps[row_indices[end - 1]].as_py())
        frame_dt = (1.0 / fps) if fps else 0.0
        return start_ts, last_ts + frame_dt
    if fps:
        return start / fps, end / fps
    return float(start), float(end)


def _action_matrix(table: pa.Table, row_indices: list[int], action_cols: list[str]) -> np.ndarray | None:
    actions = []
    for col in action_cols:
        try:
            vals = np.array([table.column(col)[i].as_py() for i in row_indices], dtype=np.float64)
        except (ValueError, TypeError):
            continue
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        elif vals.ndim > 1:
            vals = vals.reshape(len(row_indices), -1)
        else:
            continue
        actions.append(vals)
    if not actions:
        return None
    return np.concatenate(actions, axis=1)


def _apply_table_plans(table_plans: list[TableTrimPlan], fps: int, progress: Callable[[str], None] | None = None):
    next_index = 0
    frame_counters: dict[int, int] = {}
    total = len(table_plans)
    for i, plan in enumerate(table_plans, start=1):
        _emit(progress, f"Writing parquet {i}/{total}: {plan.path.name}")
        table = plan.table.take(plan.rows_to_keep) if len(plan.rows_to_keep) < len(plan.table) else plan.table
        table, next_index = _reindex_table(table, fps, next_index, frame_counters)
        pq.write_table(table, plan.path)


def _reindex_table(
    table: pa.Table,
    fps: int,
    start_index: int,
    frame_counters: dict[int, int],
) -> tuple[pa.Table, int]:
    if "episode_index" not in table.column_names:
        return table, start_index + len(table)

    ep_col = [int(ep) for ep in table.column("episode_index").to_pylist()]
    new_frames = []
    new_timestamps = []
    for ep in ep_col:
        counter = frame_counters.get(ep, 0)
        new_frames.append(counter)
        new_timestamps.append(counter / fps)
        frame_counters[ep] = counter + 1

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
    episode_plans: dict[int, EpisodeTrimPlan],
) -> tuple[dict[Path, list[VideoSlice]], dict[tuple[str, int, int], tuple[float, float]]]:
    info = _load_info(root)
    if not info:
        return {}, {}

    features = info.get("features", {})
    video_features = [name for name, spec in features.items() if spec.get("dtype") == "video"]
    if not video_features:
        return {}, {}
    if not info.get("video_path"):
        raise RuntimeError("video features are declared but video_path is missing in info.json")
    fps = info.get("fps")
    if not fps:
        raise RuntimeError("video trimming requires fps in info.json")

    meta_rows = _load_episode_meta_rows_all(root)
    if not meta_rows:
        raise RuntimeError("video trimming requires meta/episodes parquet metadata")

    rewrites: dict[Path, list[VideoSlice]] = {}
    windows: dict[tuple[str, int, int], tuple[float, float]] = {}

    for feat_name in video_features:
        feat_rewrites: dict[Path, list[VideoSlice]] = {}
        for ep_idx in sorted(episode_plans):
            plan = episode_plans[ep_idx]
            if plan.dropped:
                continue
            for shard in plan.shards:
                if shard.new_length <= 0:
                    continue
                raw = meta_rows.get((ep_idx, shard.data_file_index))
                if raw is None:
                    raise RuntimeError(
                        f"episode {ep_idx} shard data/file_index={shard.data_file_index} missing from meta/episodes"
                    )
                v_chunk = raw.get(f"videos/{feat_name}/chunk_index")
                v_file = raw.get(f"videos/{feat_name}/file_index")
                from_ts = raw.get(f"videos/{feat_name}/from_timestamp")
                if v_chunk is None or v_file is None or from_ts is None:
                    raise RuntimeError(
                        f"episode {ep_idx} shard data/file_index={shard.data_file_index} missing "
                        f"videos/{feat_name} chunk_index/file_index/from_timestamp; "
                        "rerun with --no-trim-videos to trim parquet only"
                    )

                video_path = _format_video_path(info["video_path"], feat_name, int(v_chunk), int(v_file))
                source_path = root / video_path
                if not source_path.exists():
                    raise RuntimeError(f"video file not found: {source_path}")

                start_ts = float(from_ts) + shard.kept_local_start / fps
                end_ts = float(from_ts) + shard.kept_local_end / fps
                feat_rewrites.setdefault(source_path, []).append(
                    VideoSlice(
                        episode_index=ep_idx,
                        data_file_index=shard.data_file_index,
                        source_path=source_path,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        new_length=shard.new_length,
                    )
                )

        for source_path, slices in feat_rewrites.items():
            slices.sort(key=lambda s: (s.episode_index, s.data_file_index))
            next_ts = 0.0
            for sl in slices:
                end_ts = next_ts + sl.new_length / fps
                windows[(feat_name, sl.episode_index, sl.data_file_index)] = (next_ts, end_ts)
                next_ts = end_ts
            rewrites[source_path] = slices

    return rewrites, windows


def _rewrite_video_chunks(
    video_rewrites: dict[Path, list[VideoSlice]],
    fps: int,
    workers: int,
    progress: Callable[[str], None] | None = None,
) -> str:
    """Encode each per-shard trim slice independently, verify its frame count,
    then concatenate the slices into the final chunk with the demuxer (no re-encode).
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH; rerun with --no-trim-videos to trim parquet only")
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not found on PATH; cannot verify trimmed chunks")

    encoder = _choose_ffmpeg_encoder(ffmpeg)
    use_hwaccel = _has_videotoolbox_hwaccel(ffmpeg)

    cpu_total = os.cpu_count() or workers
    effective_workers = max(1, min(workers, cpu_total))
    ffmpeg_threads = max(1, cpu_total // effective_workers)

    slice_jobs: list[tuple[Path, VideoSlice]] = []
    chunk_slices: dict[Path, list[Path]] = {}
    for source_path, slices in video_rewrites.items():
        if not slices:
            continue
        chunk_slices[source_path] = []
        for i, sl in enumerate(slices):
            slice_path = source_path.with_name(f"{source_path.stem}.slice_{i:04d}.mp4")
            slice_jobs.append((slice_path, sl))
            chunk_slices[source_path].append(slice_path)

    _emit(
        progress,
        f"Encoding {len(slice_jobs)} shard slice(s) across {len(chunk_slices)} chunk(s) "
        f"with {effective_workers} worker(s) (ffmpeg -threads {ffmpeg_threads}).",
    )

    def encode_slice(slice_path: Path, sl: VideoSlice):
        vf = (
            f"trim=start={sl.start_ts:.9f}:end={sl.end_ts:.9f},"
            f"setpts=PTS-STARTPTS"
        )
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
        if use_hwaccel:
            cmd.extend(["-hwaccel", "videotoolbox"])
        cmd.extend([
            "-i", str(sl.source_path),
            "-vf", vf,
            "-an",
            "-fps_mode", "passthrough",
            "-c:v", encoder,
            "-pix_fmt", "yuv420p",
            "-threads", str(ffmpeg_threads),
        ])
        if "hevc" in encoder or "x265" in encoder:
            cmd.extend(["-tag:v", "hvc1"])
        cmd.append(str(slice_path))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            slice_path.unlink(missing_ok=True)
            msg = e.stderr.strip() or e.stdout.strip() or str(e)
            raise RuntimeError(
                f"ffmpeg failed for episode {sl.episode_index} shard data/file_index={sl.data_file_index} "
                f"({sl.source_path.name}): {msg}"
            ) from e
        actual = _count_frames(ffprobe, slice_path)
        if actual != sl.new_length:
            slice_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"episode {sl.episode_index} shard data/file_index={sl.data_file_index} "
                f"({sl.source_path.name}) slice produced {actual} frames, expected {sl.new_length}"
            )

    started = time.monotonic()
    completed = 0
    progress_step = max(1, len(slice_jobs) // 20)
    pending_error: BaseException | None = None
    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
        futures = {pool.submit(encode_slice, sp, sl): sp for sp, sl in slice_jobs}
        for fut in as_completed(futures):
            try:
                fut.result()
            except BaseException as e:
                pending_error = pending_error or e
                continue
            completed += 1
            if completed % progress_step == 0 or completed == len(slice_jobs):
                _emit(progress, f"  Encoded {completed}/{len(slice_jobs)} slice(s)")

    if pending_error is not None:
        for slice_paths in chunk_slices.values():
            for p in slice_paths:
                p.unlink(missing_ok=True)
        raise pending_error

    total_chunks = len(chunk_slices)
    for chunk_i, (source_path, slice_paths) in enumerate(chunk_slices.items(), start=1):
        _emit(
            progress,
            f"Concatenating chunk {chunk_i}/{total_chunks}: {source_path.name} ({len(slice_paths)} slice(s))",
        )
        _concat_slices(ffmpeg, slice_paths, source_path)
        for p in slice_paths:
            p.unlink(missing_ok=True)

    elapsed = time.monotonic() - started
    _emit(progress, f"Video trimming finished in {_format_duration(elapsed)}")
    return encoder


def _concat_slices(ffmpeg: str, slice_paths: list[Path], output_path: Path):
    list_path = output_path.with_name(output_path.stem + ".concat.txt")
    tmp_out = output_path.with_name(output_path.name + ".tmp.mp4")
    list_path.write_text("".join(f"file '{p.resolve()}'\n" for p in slice_paths))
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-c", "copy",
        str(tmp_out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        tmp_out.replace(output_path)
    except subprocess.CalledProcessError as e:
        tmp_out.unlink(missing_ok=True)
        msg = e.stderr.strip() or e.stdout.strip() or str(e)
        raise RuntimeError(f"ffmpeg concat failed for {output_path.name}: {msg}") from e
    finally:
        list_path.unlink(missing_ok=True)


def _count_frames(ffprobe: str, video_path: Path) -> int:
    proc = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-count_frames", "-show_entries", "stream=nb_read_frames",
            "-of", "csv=p=0", str(video_path),
        ],
        check=True, capture_output=True, text=True,
    )
    return int(proc.stdout.strip())


def _emit(progress: Callable[[str], None] | None, message: str):
    if progress is not None:
        progress(message)


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rest = int(seconds % 60)
    return f"{minutes}m{rest:02d}s"


def _choose_ffmpeg_encoder(ffmpeg: str) -> str:
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            check=True,
            capture_output=True,
            text=True,
        )
        encoders = proc.stdout
    except Exception:
        return "libx264"

    for encoder in ("hevc_videotoolbox", "h264_videotoolbox", "libx265", "libx264"):
        if encoder in encoders:
            return encoder
    return "libx264"


def _has_videotoolbox_hwaccel(ffmpeg: str) -> bool:
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-hwaccels"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return False
    return "videotoolbox" in proc.stdout


def _update_metadata_after_trim(
    root: Path,
    episode_plans: dict[int, EpisodeTrimPlan],
    video_windows: dict[tuple[str, int, int], tuple[float, float]],
):
    _update_info_counts(root, episode_plans)
    _update_episode_metadata(root, episode_plans, video_windows)


def _update_info_counts(root: Path, episode_plans: dict[int, EpisodeTrimPlan]):
    info = _load_info(root)
    if not info:
        return
    kept = [p for p in episode_plans.values() if not p.dropped]
    info["total_episodes"] = len(kept)
    info["total_frames"] = sum(p.new_length for p in kept)
    info_path = root / "meta" / "info.json"
    info_path.write_text(json.dumps(info, indent=2))


def _update_video_codec(root: Path, encoder: str):
    info = _load_info(root)
    if not info:
        return
    changed = False
    for spec in info.get("features", {}).values():
        if spec.get("dtype") == "video":
            video_info = spec.setdefault("info", {})
            video_info["video.codec"] = "hevc" if ("hevc" in encoder or "x265" in encoder) else "h264"
            video_info["video.pix_fmt"] = "yuv420p"
            changed = True
    if changed:
        (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))


def _update_episode_metadata(
    root: Path,
    episode_plans: dict[int, EpisodeTrimPlan],
    video_windows: dict[tuple[str, int, int], tuple[float, float]],
):
    """Update each meta row in place; multi-shard episodes keep one row per shard.

    Rows for fully-dropped episodes, or shards whose trim intersection is empty,
    are removed entirely so length / dataset_from_index never lie.
    """
    meta_dir = root / "meta" / "episodes"
    if not meta_dir.exists():
        return

    shard_lookup: dict[tuple[int, int], EpisodeTrimShard] = {}
    for ep_idx, plan in episode_plans.items():
        if plan.dropped:
            continue
        for shard in plan.shards:
            if shard.new_length > 0:
                shard_lookup[(ep_idx, shard.data_file_index)] = shard

    for pf in sorted(meta_dir.rglob("*.parquet")):
        table = pq.read_table(pf)
        if "episode_index" not in table.column_names:
            continue
        ep_col = [int(ep) for ep in table.column("episode_index").to_pylist()]
        if "data/file_index" in table.column_names:
            df_col = [int(df) for df in table.column("data/file_index").to_pylist()]
        else:
            df_col = ep_col[:]

        keep_rows = [
            i for i in range(len(table))
            if ep_col[i] not in episode_plans
            or (ep_col[i], df_col[i]) in shard_lookup
        ]
        if len(keep_rows) < len(table):
            table = table.take(keep_rows) if keep_rows else table.slice(0, 0)
            if len(table):
                ep_col = [int(ep) for ep in table.column("episode_index").to_pylist()]
                df_col = (
                    [int(df) for df in table.column("data/file_index").to_pylist()]
                    if "data/file_index" in table.column_names
                    else ep_col[:]
                )
            else:
                ep_col, df_col = [], []
        if len(table) == 0:
            pq.write_table(table, pf)
            continue

        def shard_for(i: int) -> EpisodeTrimShard | None:
            return shard_lookup.get((ep_col[i], df_col[i]))

        updates: dict[str, list] = {}
        if "length" in table.column_names:
            col = table.column("length")
            updates["length"] = [
                shard_for(i).new_length if shard_for(i) is not None else col[i].as_py()
                for i in range(len(table))
            ]
        if "dataset_from_index" in table.column_names:
            col = table.column("dataset_from_index")
            updates["dataset_from_index"] = [
                shard_for(i).global_from_index if shard_for(i) is not None else col[i].as_py()
                for i in range(len(table))
            ]
        if "dataset_to_index" in table.column_names:
            col = table.column("dataset_to_index")
            vals = []
            for i in range(len(table)):
                sh = shard_for(i)
                vals.append(sh.global_from_index + sh.new_length if sh is not None else col[i].as_py())
            updates["dataset_to_index"] = vals

        for col_name in table.column_names:
            if not col_name.startswith("videos/") or not col_name.endswith(("/from_timestamp", "/to_timestamp")):
                continue
            feature = col_name.removeprefix("videos/").rsplit("/", 1)[0]
            is_from = col_name.endswith("/from_timestamp")
            col = table.column(col_name)
            vals = []
            for i in range(len(table)):
                window = video_windows.get((feature, ep_col[i], df_col[i]))
                vals.append((window[0] if is_from else window[1]) if window else col[i].as_py())
            updates[col_name] = vals

        for col_name, values in updates.items():
            table = table.set_column(table.column_names.index(col_name), col_name, pa.array(values))
        pq.write_table(table, pf)


def _load_episode_meta_rows_all(root: Path) -> dict[tuple[int, int], dict]:
    """Return every meta row keyed by (episode_index, data/file_index).

    LeRobot v2.1 episodes can span multiple data files; each (episode, data_file)
    pair has its own meta row with its own video window. The earlier single-key
    loader collapsed multi-shard episodes and silently dropped information.

    If meta has no `data/file_index` column (older or test-only datasets), fall
    back to keying by (episode_index, episode_index), matching the typical
    one-episode-per-file naming convention.
    """
    rows: dict[tuple[int, int], dict] = {}
    meta_dir = root / "meta" / "episodes"
    if not meta_dir.exists():
        return rows
    for pf in sorted(meta_dir.rglob("*.parquet")):
        table = pq.read_table(pf)
        if "episode_index" not in table.column_names:
            continue
        has_df = "data/file_index" in table.column_names
        for i in range(len(table)):
            row = {col: table.column(col)[i].as_py() for col in table.column_names}
            ep = int(row["episode_index"])
            df = int(row["data/file_index"]) if has_df else ep
            rows[(ep, df)] = row
    return rows


_DATA_PATH_PATTERN = re.compile(r"chunk-(\d+).*?file-(\d+)\.parquet$")


def _parse_data_path(pf: Path) -> tuple[int, int]:
    """Parse (chunk_index, file_index) from a data parquet path."""
    m = _DATA_PATH_PATTERN.search(str(pf))
    if not m:
        raise RuntimeError(f"could not parse chunk/file index from data path: {pf}")
    return int(m.group(1)), int(m.group(2))


def _format_video_path(template: str, feat_name: str, chunk_idx: int, file_idx: int) -> str:
    try:
        return template.format(
            video_key=feat_name,
            episode_chunk=chunk_idx,
            episode_index=file_idx,
            chunk_index=chunk_idx,
            file_index=file_idx,
        )
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"could not resolve video_path template: {template}") from e


def _load_info(root: Path) -> dict | None:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return None
    return json.loads(info_path.read_text())


def _load_fps(root: Path) -> int | None:
    info = _load_info(root)
    fps = info.get("fps") if info else None
    return int(fps) if fps else None
