"""Dataset subsampling -- lower LeRobot capture FPS by keeping an even frame cadence."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot_doctor.trim import (
    _choose_ffmpeg_encoder,
    _concat_slices,
    _count_frames,
    _format_duration,
    _format_video_path,
    _has_videotoolbox_hwaccel,
    _load_episode_meta_rows_all,
    _load_fps,
    _load_info,
    _parse_data_path,
)


@dataclass
class SubsampleResult:
    source_fps: int | None
    target_fps: int
    frames_removed: int
    details: list[str]


@dataclass
class EpisodeShard:
    """One (episode, data_file) pair. Episodes can have multiple shards in LeRobot v2.1.

    start_offset is the cumulative number of episode rows in earlier shards; it sets
    the subsample cadence so frame selection stays continuous across shard boundaries.
    """
    episode_index: int
    data_chunk_index: int
    data_file_index: int
    start_offset: int
    old_length: int
    new_length: int
    global_from_index: int = 0  # post-subsample row index where this shard begins


@dataclass
class EpisodeSubsamplePlan:
    episode_index: int
    shards: list[EpisodeShard] = field(default_factory=list)

    @property
    def old_length(self) -> int:
        return sum(s.old_length for s in self.shards)

    @property
    def new_length(self) -> int:
        return sum(s.new_length for s in self.shards)


@dataclass
class TableSubsamplePlan:
    path: Path
    table: pa.Table
    rows_to_keep: list[int]


@dataclass
class VideoSubsampleSlice:
    """One shard rendered as one ffmpeg invocation.

    start_offset is the episode-global row offset before this shard so the
    `select` filter can keep the same n,step cadence the parquet uses.
    """
    episode_index: int
    data_file_index: int
    source_path: Path
    start_frame: int
    old_length: int
    new_length: int
    start_offset: int


def subsample_dataset(
    root: Path,
    target_fps: int,
    subsample_videos: bool = True,
    dry_run: bool = False,
    workers: int = 4,
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
            encoder = _rewrite_video_chunks(
                video_rewrites, source_fps, target_fps, workers=workers, progress=progress
            )
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

    step = source_fps // target_fps
    episode_offsets: dict[int, int] = {}
    cursor = 0  # post-subsample global row index

    for pf in sorted(data_dir.rglob("*.parquet")):
        table = pq.read_table(pf)
        if "episode_index" not in table.column_names:
            table_plans.append(TableSubsamplePlan(pf, table, list(range(len(table)))))
            cursor += len(table)
            continue

        chunk_idx, file_idx = _parse_data_path(pf)

        ep_col = table.column("episode_index").to_pylist()
        episodes: dict[int, list[int]] = {}
        for i, ep in enumerate(ep_col):
            episodes.setdefault(int(ep), []).append(i)

        rows_to_keep: list[int] = []
        for ep_idx in sorted(episodes.keys()):
            row_indices = episodes[ep_idx]
            start_offset = episode_offsets.get(ep_idx, 0)
            kept_rows = [
                row
                for local_offset, row in enumerate(row_indices)
                if (start_offset + local_offset) % step == 0
            ]
            episode_offsets[ep_idx] = start_offset + len(row_indices)
            rows_to_keep.extend(kept_rows)

            shard = EpisodeShard(
                episode_index=ep_idx,
                data_chunk_index=chunk_idx,
                data_file_index=file_idx,
                start_offset=start_offset,
                old_length=len(row_indices),
                new_length=len(kept_rows),
                global_from_index=cursor,
            )
            cursor += len(kept_rows)
            episode_plans.setdefault(ep_idx, EpisodeSubsamplePlan(ep_idx)).shards.append(shard)
            result.frames_removed += len(row_indices) - len(kept_rows)

        table_plans.append(TableSubsamplePlan(pf, table, rows_to_keep))

    for ep_idx in sorted(episode_plans):
        plan = episode_plans[ep_idx]
        if plan.old_length != plan.new_length:
            shard_note = f" across {len(plan.shards)} shards" if len(plan.shards) > 1 else ""
            result.details.append(
                f"Episode {ep_idx}: subsampled {plan.old_length} -> {plan.new_length} frames "
                f"({source_fps}fps -> {target_fps}fps){shard_note}"
            )

    return table_plans, episode_plans


def _apply_table_plans(
    table_plans: list[TableSubsamplePlan],
    target_fps: int,
    progress: Callable[[str], None] | None = None,
):
    next_index = 0
    frame_counters: dict[int, int] = {}
    total = len(table_plans)
    for i, plan in enumerate(table_plans, start=1):
        _emit(progress, f"Writing subsampled parquet {i}/{total}: {plan.path.name}")
        table = plan.table.take(plan.rows_to_keep) if len(plan.rows_to_keep) < len(plan.table) else plan.table
        table, next_index = _reindex_table(table, target_fps, next_index, frame_counters)
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
    episode_plans: dict[int, EpisodeSubsamplePlan],
    source_fps: int,
    target_fps: int,
) -> tuple[dict[Path, list[VideoSubsampleSlice]], dict[tuple[str, int, int], tuple[float, float]]]:
    info = _load_info(root)
    if not info:
        return {}, {}

    features = info.get("features", {})
    video_features = [name for name, spec in features.items() if spec.get("dtype") == "video"]
    if not video_features:
        return {}, {}
    if not info.get("video_path"):
        raise RuntimeError("video features are declared but video_path is missing in info.json")

    meta_rows = _load_episode_meta_rows_all(root)
    if not meta_rows:
        raise RuntimeError("video subsampling requires meta/episodes parquet metadata")

    rewrites: dict[Path, list[VideoSubsampleSlice]] = {}
    windows: dict[tuple[str, int, int], tuple[float, float]] = {}

    for feat_name in video_features:
        feat_rewrites: dict[Path, list[VideoSubsampleSlice]] = {}
        for ep_idx in sorted(episode_plans):
            plan = episode_plans[ep_idx]
            for shard in plan.shards:
                key = (ep_idx, shard.data_file_index)
                raw = meta_rows.get(key)
                if raw is None:
                    raise RuntimeError(
                        f"episode {ep_idx} shard data/file_index={shard.data_file_index} missing from meta/episodes"
                    )
                v_chunk = raw.get(f"videos/{feat_name}/chunk_index")
                v_file = raw.get(f"videos/{feat_name}/file_index")
                from_ts = raw.get(f"videos/{feat_name}/from_timestamp")
                to_ts = raw.get(f"videos/{feat_name}/to_timestamp")
                if v_chunk is None or v_file is None or from_ts is None or to_ts is None:
                    raise RuntimeError(
                        f"episode {ep_idx} shard data/file_index={shard.data_file_index} missing "
                        f"videos/{feat_name} chunk_index/file_index/from_timestamp/to_timestamp"
                    )
                start_frame = int(round(float(from_ts) * source_fps))
                end_frame = int(round(float(to_ts) * source_fps))
                if end_frame - start_frame != shard.old_length:
                    raise RuntimeError(
                        f"episode {ep_idx} shard data/file_index={shard.data_file_index} "
                        f"videos/{feat_name} window has {end_frame - start_frame} frames "
                        f"(from_ts={from_ts}, to_ts={to_ts}, source_fps={source_fps}) but "
                        f"parquet shard has {shard.old_length} rows"
                    )

                video_path = _format_video_path(info["video_path"], feat_name, int(v_chunk), int(v_file))
                source_path = root / video_path
                if not source_path.exists():
                    raise RuntimeError(f"video file not found: {source_path}")

                feat_rewrites.setdefault(source_path, []).append(
                    VideoSubsampleSlice(
                        episode_index=ep_idx,
                        data_file_index=shard.data_file_index,
                        source_path=source_path,
                        start_frame=start_frame,
                        old_length=shard.old_length,
                        new_length=shard.new_length,
                        start_offset=shard.start_offset,
                    )
                )

        for source_path, slices in feat_rewrites.items():
            slices.sort(key=lambda s: (s.episode_index, s.data_file_index))
            next_ts = 0.0
            for sl in slices:
                end_ts = next_ts + sl.new_length / target_fps
                windows[(feat_name, sl.episode_index, sl.data_file_index)] = (next_ts, end_ts)
                next_ts = end_ts
            rewrites[source_path] = slices

    return rewrites, windows


def _rewrite_video_chunks(
    video_rewrites: dict[Path, list[VideoSubsampleSlice]],
    source_fps: int,
    target_fps: int,
    workers: int,
    progress: Callable[[str], None] | None = None,
) -> str:
    """Encode each shard slice independently, verify its frame count, then
    concatenate the slices into the final chunk with the demuxer (no re-encode).

    Per-slice encoding makes ffmpeg's output frame count deterministic and locally
    checkable; demuxer concat with -c copy can never drop frames.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH; cannot subsample videos")
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not found on PATH; cannot verify subsampled chunks")

    encoder = _choose_ffmpeg_encoder(ffmpeg)
    use_hwaccel = _has_videotoolbox_hwaccel(ffmpeg)

    cpu_total = os.cpu_count() or workers
    effective_workers = max(1, min(workers, cpu_total))
    # libx265 rejects frame-threads >= 16 (X265_MAX_FRAME_THREADS); cap to stay safe
    # across both x264 and x265 while still benefitting from intra-ffmpeg parallelism.
    ffmpeg_threads = max(1, min(cpu_total // effective_workers, 8))

    step = source_fps // target_fps

    slice_jobs: list[tuple[Path, VideoSubsampleSlice]] = []
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

    def encode_slice(slice_path: Path, sl: VideoSubsampleSlice):
        end_frame = sl.start_frame + sl.old_length
        # Keep frames where (n + start_offset) % step == 0 so the cadence stays
        # continuous when an episode spans multiple shards.
        offset = sl.start_offset % step
        select_expr_local = f"eq(mod(n+{offset}\\,{step})\\,0)"
        vf = (
            f"trim=start_frame={sl.start_frame}:end_frame={end_frame},"
            f"setpts=PTS-STARTPTS,select='{select_expr_local}',"
            f"setpts=N/({target_fps}*TB)"
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
            "-bf", "0",
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
    _emit(progress, f"Video subsampling finished in {_format_duration(elapsed)}")
    return encoder


def _update_metadata_after_subsample(
    root: Path,
    episode_plans: dict[int, EpisodeSubsamplePlan],
    video_windows: dict[tuple[str, int, int], tuple[float, float]],
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
    video_windows: dict[tuple[str, int, int], tuple[float, float]],
):
    """Update each meta row in place; multi-shard episodes keep one row per shard."""
    meta_dir = root / "meta" / "episodes"
    if not meta_dir.exists():
        return

    shard_lookup: dict[tuple[int, int], EpisodeShard] = {}
    for plan in episode_plans.values():
        for shard in plan.shards:
            shard_lookup[(shard.episode_index, shard.data_file_index)] = shard

    for pf in sorted(meta_dir.rglob("*.parquet")):
        table = pq.read_table(pf)
        if "episode_index" not in table.column_names:
            continue
        ep_col = [int(ep) for ep in table.column("episode_index").to_pylist()]
        if "data/file_index" in table.column_names:
            df_col = [int(df) for df in table.column("data/file_index").to_pylist()]
        else:
            df_col = ep_col[:]
        if len(table) == 0:
            continue

        updates: dict[str, list] = {}

        def shard_for(i: int) -> EpisodeShard | None:
            return shard_lookup.get((ep_col[i], df_col[i]))

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


def _emit(progress: Callable[[str], None] | None, message: str):
    if progress is not None:
        progress(message)
