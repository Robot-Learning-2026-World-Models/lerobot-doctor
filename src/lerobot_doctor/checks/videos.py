"""Check 4: Video Integrity."""

from __future__ import annotations

from pathlib import Path

from lerobot_doctor.dataset_loader import LoadedDataset
from lerobot_doctor.runner import CheckResult, Severity


def check_videos(dataset: LoadedDataset) -> CheckResult:
    result = CheckResult(name="Video Integrity", severity=Severity.PASS)

    if dataset.info is None:
        result.fail("Cannot check videos: info.json not loaded")
        return result

    # Find video features
    video_features = {
        name: spec for name, spec in dataset.info.features.items()
        if spec.get("dtype") == "video"
    }

    if not video_features:
        result.pass_("No video features declared -- skipping video checks")
        return result

    video_path_template = dataset.info.video_path
    if not video_path_template:
        result.fail("video_path not set in info.json but video features declared")
        return result

    if not dataset.is_local:
        result.warn(
            f"Found {len(video_features)} video feature(s): {list(video_features.keys())} "
            f"-- skipping video decode checks for remote dataset (videos not downloaded)"
        )
        return result

    result.pass_(f"Found {len(video_features)} video feature(s): {list(video_features.keys())}")

    for feat_name, feat_spec in video_features.items():
        _check_video_feature(dataset, feat_name, feat_spec, video_path_template, result)

    return result


def _check_video_feature(
    dataset: LoadedDataset,
    feat_name: str,
    feat_spec: dict,
    video_path_template: str,
    result: CheckResult,
):
    root = dataset.root
    fps = dataset.info.fps if dataset.info else None
    expected_shape = feat_spec.get("shape")  # e.g. [3, 480, 640] or [480, 640, 3]

    # Check video files exist for each episode
    missing_videos = []
    checked = 0
    decode_errors = []
    fps_mismatches = []
    resolution_mismatches = []
    frame_count_mismatches = []
    videos_by_path = {}

    for ep_meta in dataset.episodes_meta:
        ep_idx = ep_meta.episode_index
        chunks_size = dataset.info.chunks_size or 1000

        # Get chunk/file indices from episode metadata if available, else compute
        vid_chunk_key = f"videos/{feat_name}/chunk_index"
        vid_file_key = f"videos/{feat_name}/file_index"
        chunk_idx = ep_meta.raw.get(vid_chunk_key, ep_idx // chunks_size)
        file_idx = ep_meta.raw.get(vid_file_key, ep_idx)

        # Build video path from template -- handle different template variable names
        try:
            video_path = video_path_template.format(
                video_key=feat_name,
                episode_chunk=chunk_idx,
                episode_index=file_idx,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )
        except (KeyError, IndexError):
            # Unknown template variables -- skip this feature
            result.warn(f"{feat_name}: Could not resolve video_path template: {video_path_template}")
            return

        full_path = root / video_path

        if not full_path.exists():
            missing_videos.append(ep_idx)
            continue

        from_ts = ep_meta.raw.get(f"videos/{feat_name}/from_timestamp")
        to_ts = ep_meta.raw.get(f"videos/{feat_name}/to_timestamp")
        videos_by_path.setdefault(full_path, []).append((ep_idx, from_ts, to_ts))

    for full_path, episode_windows in videos_by_path.items():
        checked += 1

        # Only do detailed checks on a sample to avoid being too slow
        if checked > 20:
            continue

        try:
            import av
            container = av.open(str(full_path))
            stream = container.streams.video[0]
            sample_ep = episode_windows[0][0]

            # Check fps
            if fps and stream.average_rate:
                video_fps = float(stream.average_rate)
                if abs(video_fps - fps) > 1.0:
                    fps_mismatches.append((sample_ep, video_fps))

            # Check resolution
            if expected_shape and stream.width and stream.height:
                # Shape could be [C, H, W] or [H, W, C]
                h, w = stream.height, stream.width
                shape_matches = False
                for perm in [expected_shape, list(reversed(expected_shape))]:
                    if len(perm) >= 2:
                        if (h in perm and w in perm):
                            shape_matches = True
                            break
                if not shape_matches:
                    resolution_mismatches.append((sample_ep, h, w, expected_shape))

            # Check frame count vs episode metadata. LeRobot v3 commonly stores
            # many episode windows in one larger MP4 chunk, so compare the
            # whole chunk against the sum of its episode windows.
            if fps:
                complete_windows = [
                    (ep_idx, from_ts, to_ts)
                    for ep_idx, from_ts, to_ts in episode_windows
                    if from_ts is not None and to_ts is not None
                ]
                if complete_windows:
                    expected_frames = round(sum((to_ts - from_ts) * fps for _, from_ts, to_ts in complete_windows))
                    # Count actual frames
                    actual_frames = stream.frames
                    if actual_frames == 0:
                        # Some containers don't report frame count, decode to count
                        container.seek(0)
                        actual_frames = sum(1 for _ in container.decode(video=0))
                    if actual_frames > 0 and abs(actual_frames - expected_frames) > 2:
                        first_ep = complete_windows[0][0]
                        last_ep = complete_windows[-1][0]
                        frame_count_mismatches.append((first_ep, last_ep, full_path.name, actual_frames, expected_frames))

            # Try decoding first frame to check for corruption
            container.seek(0)
            for frame in container.decode(video=0):
                break  # just need first frame
            else:
                decode_errors.append(sample_ep)

            container.close()
        except ImportError:
            result.warn("PyAV (av) not installed -- skipping video decode checks")
            return
        except Exception as e:
            decode_errors.append(episode_windows[0][0])

    if missing_videos:
        if len(missing_videos) > 10:
            result.fail(
                f"{feat_name}: {len(missing_videos)} video files missing "
                f"(episodes {missing_videos[:5]}...)"
            )
        else:
            result.fail(f"{feat_name}: Video files missing for episodes {missing_videos}")
    else:
        result.pass_(f"{feat_name}: All video files present")

    if decode_errors:
        result.fail(f"{feat_name}: {len(decode_errors)} video(s) failed to decode: episodes {decode_errors[:5]}")

    if fps_mismatches:
        for ep_idx, vfps in fps_mismatches[:3]:
            result.warn(f"{feat_name}: Episode {ep_idx} video fps={vfps:.1f} != dataset fps={fps}")

    if resolution_mismatches:
        for ep_idx, h, w, expected in resolution_mismatches[:3]:
            result.warn(f"{feat_name}: Episode {ep_idx} resolution {w}x{h} doesn't match shape {expected}")

    if frame_count_mismatches:
        for first_ep, last_ep, filename, actual, expected in frame_count_mismatches[:3]:
            if first_ep == last_ep:
                result.warn(f"{feat_name}: Episode {first_ep} video {filename} has {actual} frames, expected {expected}")
            else:
                result.warn(
                    f"{feat_name}: Video chunk {filename} for episodes {first_ep}-{last_ep} "
                    f"has {actual} frames, expected {expected}"
                )
