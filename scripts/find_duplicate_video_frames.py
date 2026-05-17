#!/usr/bin/env python3
"""Find bit-identical duplicate video frames in a local LeRobot dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download


@dataclass
class EpisodeWindow:
    feature: str
    episode_index: int
    video_path: Path
    start_frame: int
    end_frame: int
    expected_length: int


@dataclass
class DuplicateRun:
    feature: str
    episode_index: int
    start_frame_index: int
    end_frame_index: int
    length: int


@dataclass
class EpisodeStats:
    feature: str
    episode_index: int
    frames_seen: int = 0
    duplicate_frames: int = 0
    duplicate_runs: list[DuplicateRun] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Traverse local LeRobot video episodes frame-by-frame, convert frames "
            "to uint8 numpy arrays, and report consecutive bit-identical frames "
            "using XOR diffs."
        )
    )
    parser.add_argument("dataset", help="Local dataset root or Hugging Face dataset repo id, e.g. org/name")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional Hugging Face cache directory")
    parser.add_argument("--features", nargs="+", default=None, help="Video feature names to inspect; default all")
    parser.add_argument("--max-episodes", type=int, default=None, help="Only inspect first N episode indices")
    parser.add_argument("--max-examples", type=int, default=200, help="Max duplicate runs printed to terminal")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON report path")
    args = parser.parse_args(argv)

    root = _resolve_dataset_root(args.dataset, args.cache_dir)

    info = _load_info(root)
    fps = info.get("fps")
    if not fps:
        raise SystemExit("meta/info.json fps is required")
    fps = int(fps)

    video_features = _video_features(info, args.features)
    if not video_features:
        raise SystemExit("No matching video features found in meta/info.json")

    windows_by_path = _build_windows(root, info, video_features, fps, args.max_episodes)
    if not windows_by_path:
        raise SystemExit("No episode video windows found")

    print(f"Dataset: {root}")
    print(f"FPS: {fps}")
    print(f"Features: {', '.join(video_features)}")
    print(f"Video chunks: {len(windows_by_path)}")
    print()

    stats = _scan_windows(windows_by_path)
    report = _summarize(stats)
    _print_report(report, args.max_examples)

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(report, indent=2))
        print(f"\nJSON report written: {args.json_out}")

    return 0


def _resolve_dataset_root(dataset: str, cache_dir: Path | None) -> Path:
    local = Path(dataset).expanduser()
    if local.exists():
        return local.resolve()
    if "/" not in dataset:
        raise SystemExit(f"Dataset path not found and not a repo id: {dataset}")

    print(f"Downloading Hugging Face dataset snapshot: {dataset}")
    snapshot_path = snapshot_download(
        repo_id=dataset,
        repo_type="dataset",
        cache_dir=str(cache_dir.expanduser()) if cache_dir is not None else None,
        allow_patterns=["meta/**", "videos/**"],
    )
    return Path(snapshot_path).resolve()


def _load_info(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise SystemExit(f"meta/info.json not found: {info_path}")
    return json.loads(info_path.read_text())


def _video_features(info: dict[str, Any], requested: list[str] | None) -> list[str]:
    found = [name for name, spec in info.get("features", {}).items() if spec.get("dtype") == "video"]
    if requested is None:
        return found
    requested_set = set(requested)
    missing = sorted(requested_set - set(found))
    if missing:
        raise SystemExit(f"Requested video feature(s) not found: {missing}; available: {found}")
    return [name for name in found if name in requested_set]


def _build_windows(
    root: Path,
    info: dict[str, Any],
    video_features: list[str],
    fps: int,
    max_episodes: int | None,
) -> dict[Path, list[EpisodeWindow]]:
    video_template = info.get("video_path")
    if not video_template:
        raise SystemExit("meta/info.json video_path is required")

    chunks_size = int(info.get("chunks_size", 1000))
    windows_by_path: dict[Path, list[EpisodeWindow]] = {}

    for row in _episode_rows(root):
        ep_idx = int(row.get("episode_index", -1))
        if ep_idx < 0:
            continue
        if max_episodes is not None and ep_idx >= max_episodes:
            continue
        length = int(row.get("length", 0))
        if length <= 0:
            continue

        for feature in video_features:
            chunk_idx = int(row.get(f"videos/{feature}/chunk_index", ep_idx // chunks_size))
            file_idx = int(row.get(f"videos/{feature}/file_index", ep_idx))
            from_ts = row.get(f"videos/{feature}/from_timestamp")
            if from_ts is None:
                raise SystemExit(f"episode {ep_idx} missing videos/{feature}/from_timestamp")

            rel_path = _format_video_path(video_template, feature, chunk_idx, file_idx)
            video_path = root / rel_path
            if not video_path.exists():
                raise SystemExit(f"Video file not found: {video_path}")

            start_frame = int(round(float(from_ts) * fps))
            end_frame = start_frame + length
            windows_by_path.setdefault(video_path, []).append(
                EpisodeWindow(
                    feature=feature,
                    episode_index=ep_idx,
                    video_path=video_path,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    expected_length=length,
                )
            )

    for windows in windows_by_path.values():
        windows.sort(key=lambda w: (w.start_frame, w.end_frame, w.feature, w.episode_index))
    return windows_by_path


def _episode_rows(root: Path) -> list[dict[str, Any]]:
    meta_dir = root / "meta" / "episodes"
    if not meta_dir.exists():
        raise SystemExit(f"meta/episodes not found: {meta_dir}")

    rows: list[dict[str, Any]] = []
    for parquet_path in sorted(meta_dir.rglob("*.parquet")):
        table = pq.read_table(parquet_path)
        for i in range(len(table)):
            rows.append({col: table.column(col)[i].as_py() for col in table.column_names})
    rows.sort(key=lambda row: int(row.get("episode_index", -1)))
    return rows


def _format_video_path(template: str, feature: str, chunk_idx: int, file_idx: int) -> str:
    try:
        return template.format(
            video_key=feature,
            episode_chunk=chunk_idx,
            episode_index=file_idx,
            chunk_index=chunk_idx,
            file_index=file_idx,
        )
    except (KeyError, IndexError) as exc:
        raise SystemExit(f"Could not resolve video_path template: {template}") from exc


def _scan_windows(windows_by_path: dict[Path, list[EpisodeWindow]]) -> list[EpisodeStats]:
    try:
        import av
    except ImportError as exc:
        raise SystemExit("PyAV (av) is required to decode videos") from exc

    all_stats: list[EpisodeStats] = []
    total_chunks = len(windows_by_path)

    for chunk_i, (video_path, windows) in enumerate(sorted(windows_by_path.items()), start=1):
        print(f"[{chunk_i}/{total_chunks}] scanning {video_path.name} ({len(windows)} episode window(s))")
        container = av.open(str(video_path))
        try:
            stats_by_key = {(w.feature, w.episode_index): EpisodeStats(w.feature, w.episode_index) for w in windows}
            window_iter = iter(windows)
            current = next(window_iter, None)
            next_window = next(window_iter, None)
            chunk_frame_index = -1

            for frame in container.decode(video=0):
                chunk_frame_index += 1
                while current is not None and chunk_frame_index >= current.end_frame:
                    _finish_window(stats_by_key[(current.feature, current.episode_index)], current)
                    current = next_window
                    next_window = next(window_iter, None)

                while current is not None and chunk_frame_index < current.start_frame:
                    break

                if current is None:
                    break
                if chunk_frame_index < current.start_frame:
                    continue

                arr = frame.to_ndarray(format="rgb24")
                _consume_frame(stats_by_key[(current.feature, current.episode_index)], arr)

            while current is not None:
                _finish_window(stats_by_key[(current.feature, current.episode_index)], current)
                current = next_window
                next_window = next(window_iter, None)

            all_stats.extend(stats_by_key.values())
        finally:
            container.close()

    all_stats.sort(key=lambda s: (s.feature, s.episode_index))
    return all_stats


def _consume_frame(stats: EpisodeStats, arr: np.ndarray):
    arr = np.asarray(arr, dtype=np.uint8)
    prev = getattr(stats, "_prev", None)
    duplicate = False
    if prev is not None:
        duplicate = not bool(np.bitwise_xor(prev, arr).any())

    frame_index = stats.frames_seen
    if duplicate:
        stats.duplicate_frames += 1
        if stats.duplicate_runs and stats.duplicate_runs[-1].end_frame_index == frame_index - 1:
            run = stats.duplicate_runs[-1]
            run.end_frame_index = frame_index
            run.length += 1
        else:
            stats.duplicate_runs.append(
                DuplicateRun(
                    feature=stats.feature,
                    episode_index=stats.episode_index,
                    start_frame_index=frame_index - 1,
                    end_frame_index=frame_index,
                    length=2,
                )
            )

    setattr(stats, "_prev", arr.copy())
    stats.frames_seen += 1


def _finish_window(stats: EpisodeStats, window: EpisodeWindow):
    if stats.frames_seen != window.expected_length:
        stats.warnings.append(
            f"decoded {stats.frames_seen} frame(s), expected {window.expected_length} "
            f"from chunk frames [{window.start_frame}, {window.end_frame})"
        )
    if hasattr(stats, "_prev"):
        delattr(stats, "_prev")


def _summarize(stats: list[EpisodeStats]) -> dict[str, Any]:
    duplicate_runs = [run for item in stats for run in item.duplicate_runs]
    warnings = [
        {"feature": item.feature, "episode_index": item.episode_index, "warning": warning}
        for item in stats
        for warning in item.warnings
    ]
    return {
        "episodes_scanned": len(stats),
        "frames_scanned": sum(item.frames_seen for item in stats),
        "duplicate_frames": sum(item.duplicate_frames for item in stats),
        "episodes_with_duplicates": sum(1 for item in stats if item.duplicate_frames > 0),
        "duplicate_runs": [run.__dict__ for run in duplicate_runs],
        "warnings": warnings,
        "per_episode": [
            {
                "feature": item.feature,
                "episode_index": item.episode_index,
                "frames_seen": item.frames_seen,
                "duplicate_frames": item.duplicate_frames,
                "duplicate_runs": len(item.duplicate_runs),
                "warnings": item.warnings,
            }
            for item in stats
        ],
    }


def _print_report(report: dict[str, Any], max_examples: int):
    print()
    print("Summary")
    print(f"  Episode windows scanned: {report['episodes_scanned']}")
    print(f"  Frames scanned:          {report['frames_scanned']}")
    print(f"  Duplicate frames:       {report['duplicate_frames']}")
    print(f"  Episodes with dupes:    {report['episodes_with_duplicates']}")
    print(f"  Warnings:               {len(report['warnings'])}")

    runs = report["duplicate_runs"]
    if runs:
        print()
        print(f"Duplicate runs (first {min(max_examples, len(runs))}/{len(runs)})")
        for run in runs[:max_examples]:
            print(
                "  "
                f"{run['feature']} episode={run['episode_index']} "
                f"frames={run['start_frame_index']}..{run['end_frame_index']} "
                f"run_length={run['length']}"
            )

    if report["warnings"]:
        print()
        print("Warnings")
        for warning in report["warnings"][:max_examples]:
            print(f"  {warning['feature']} episode={warning['episode_index']}: {warning['warning']}")


if __name__ == "__main__":
    raise SystemExit(main())
