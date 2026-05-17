"""Episode trimming -- remove idle/static frames from episodes."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
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


@dataclass
class TableTrimPlan:
    path: Path
    table: pa.Table
    rows_to_keep: list[int]


@dataclass
class VideoSlice:
    episode_index: int
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
            encoder = _rewrite_video_chunks(video_rewrites, fps, progress=progress)
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

    table_plans = [
        TableTrimPlan(path, table, sorted(rows_to_keep))
        for (path, table), rows_to_keep in zip(tables, table_rows_to_keep)
    ]
    return table_plans, episode_plans


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
) -> tuple[dict[Path, list[VideoSlice]], dict[tuple[str, int], tuple[float, float]]]:
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

    meta_rows = _load_episode_meta_rows(root)
    if not meta_rows:
        raise RuntimeError("video trimming requires meta/episodes parquet metadata")

    chunks_size = info.get("chunks_size", 1000)
    rewrites: dict[Path, list[VideoSlice]] = {}
    windows: dict[tuple[str, int], tuple[float, float]] = {}

    for feat_name in video_features:
        grouped: dict[Path, list[VideoSlice]] = {}
        for ep_idx in sorted(episode_plans):
            plan = episode_plans[ep_idx]
            if plan.dropped:
                continue
            raw = meta_rows.get(ep_idx)
            if raw is None:
                raise RuntimeError(f"episode {ep_idx} is missing from meta/episodes")

            chunk_idx = raw.get(f"videos/{feat_name}/chunk_index", ep_idx // chunks_size)
            file_idx = raw.get(f"videos/{feat_name}/file_index", ep_idx)
            from_ts = raw.get(f"videos/{feat_name}/from_timestamp")
            if from_ts is None:
                raise RuntimeError(
                    f"episode {ep_idx} is missing videos/{feat_name}/from_timestamp; "
                    "rerun with --no-trim-videos to trim parquet only"
                )

            video_path = _format_video_path(info["video_path"], feat_name, int(chunk_idx), int(file_idx))
            source_path = root / video_path
            if not source_path.exists():
                raise RuntimeError(f"video file not found: {source_path}")

            start_ts = float(from_ts) + plan.source_start_ts
            end_ts = float(from_ts) + plan.source_end_ts
            grouped.setdefault(source_path, []).append(
                VideoSlice(ep_idx, source_path, start_ts, end_ts, plan.new_length)
            )

        for source_path, slices in grouped.items():
            next_ts = 0.0
            rewrites[source_path] = slices
            for sl in slices:
                end_ts = next_ts + sl.new_length / fps
                windows[(feat_name, sl.episode_index)] = (next_ts, end_ts)
                next_ts = end_ts

    return rewrites, windows


def _rewrite_video_chunks(
    video_rewrites: dict[Path, list[VideoSlice]],
    fps: int,
    progress: Callable[[str], None] | None = None,
) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH; rerun with --no-trim-videos to trim parquet only")

    encoder = _choose_ffmpeg_encoder(ffmpeg)
    use_hwaccel = _has_videotoolbox_hwaccel(ffmpeg)

    total = len(video_rewrites)
    for chunk_i, (source_path, slices) in enumerate(video_rewrites.items(), start=1):
        if not slices:
            continue
        kept_frames = sum(sl.new_length for sl in slices)
        source_mb = source_path.stat().st_size / (1024 * 1024) if source_path.exists() else 0.0
        _emit(
            progress,
            f"Rewriting video chunk {chunk_i}/{total}: {source_path.name} "
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
                f"setpts=PTS-STARTPTS[{label}]"
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
            "-r", str(fps),
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
    video_windows: dict[tuple[str, int], tuple[float, float]],
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
    video_windows: dict[tuple[str, int], tuple[float, float]],
):
    meta_dir = root / "meta" / "episodes"
    if not meta_dir.exists():
        return

    global_from: dict[int, int] = {}
    cursor = 0
    for ep_idx in sorted(episode_plans):
        plan = episode_plans[ep_idx]
        if plan.dropped:
            continue
        global_from[ep_idx] = cursor
        cursor += plan.new_length

    seen_episodes: set[int] = set()
    for pf in sorted(meta_dir.rglob("*.parquet")):
        table = pq.read_table(pf)
        if "episode_index" not in table.column_names:
            continue
        ep_col = [int(ep) for ep in table.column("episode_index").to_pylist()]
        keep_rows = []
        for i, ep in enumerate(ep_col):
            if ep in seen_episodes:
                continue
            if ep in episode_plans and episode_plans[ep].dropped:
                seen_episodes.add(ep)
                continue
            keep_rows.append(i)
            seen_episodes.add(ep)
        if len(keep_rows) < len(table):
            table = table.take(keep_rows)
            ep_col = [int(ep) for ep in table.column("episode_index").to_pylist()]
        if len(table) == 0:
            pq.write_table(table, pf)
            continue

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
                global_from[ep] + episode_plans[ep].new_length if ep in episode_plans else table.column("dataset_to_index")[i].as_py()
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


def _load_episode_meta_rows(root: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    meta_dir = root / "meta" / "episodes"
    if not meta_dir.exists():
        return rows
    for pf in sorted(meta_dir.rglob("*.parquet")):
        table = pq.read_table(pf)
        for i in range(len(table)):
            row = {col: table.column(col)[i].as_py() for col in table.column_names}
            if "episode_index" in row:
                rows[int(row["episode_index"])] = row
    return rows


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
