#!/usr/bin/env python3
"""Download, trim, validate, and upload a Hugging Face LeRobot dataset."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from lerobot_doctor.dataset_loader import load_local
from lerobot_doctor.report import print_report
from lerobot_doctor.runner import Severity, run_checks
from lerobot_doctor.subsample import subsample_dataset
from lerobot_doctor.trim import trim_dataset


CHECKS = ["metadata", "temporal", "episodes", "videos"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download a Hugging Face LeRobot dataset with videos, copy it to "
            "~/Downloads, trim idle frames/videos, optionally subsample FPS, "
            "run health checks, and upload to the same namespace with suffixes "
            "appended."
        )
    )
    parser.add_argument("repo_id", help="Source dataset repo id, e.g. org/name")
    parser.add_argument("--output-parent", default=str(Path.home() / "Downloads"))
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--min-active", type=int, default=10)
    parser.add_argument("--subsample-fps", type=int, default=None, help="Optional target FPS; must divide source FPS")
    parser.add_argument("--private", action="store_true", help="Create the target Hub repo as private")
    parser.add_argument("--overwrite-local", action="store_true", help="Replace an existing local Downloads copy")
    parser.add_argument("--skip-upload", action="store_true", help="Stop after trimming and checks")
    parser.add_argument("--upload-delete-existing", action="store_true", help="Delete existing files in the target repo")
    args = parser.parse_args(argv)

    source_repo = args.repo_id
    target_repo = _trimmed_repo_id(source_repo, subsample_fps=args.subsample_fps)
    local_dir = Path(args.output_parent).expanduser() / target_repo.split("/", 1)[1]

    total_steps = 6 if args.subsample_fps is not None else 5

    print(f"Source repo: {source_repo}")
    print(f"Target repo: {target_repo}")
    print(f"Local copy:  {local_dir}")

    print(f"\n[1/{total_steps}] Downloading dataset snapshot, including videos...")
    snapshot_path = Path(snapshot_download(
        repo_id=source_repo,
        repo_type="dataset",
        allow_patterns=["meta/**", "data/**", "videos/**"],
    ))
    print(f"Downloaded snapshot: {snapshot_path}")

    print(f"\n[2/{total_steps}] Copying snapshot to local editable directory...")
    if local_dir.exists():
        if not args.overwrite_local:
            print(
                f"Local directory already exists: {local_dir}\n"
                "Rerun with --overwrite-local to replace it.",
                file=sys.stderr,
            )
            return 2
        shutil.rmtree(local_dir)
    shutil.copytree(snapshot_path, local_dir, symlinks=False)
    print(f"Copied to: {local_dir}")

    print(f"\n[3/{total_steps}] Trimming parquet rows and video chunks...")
    trim_result = trim_dataset(
        local_dir,
        action_threshold=args.threshold,
        min_active_frames=args.min_active,
        progress=lambda msg: print(f"  {msg}"),
    )
    for detail in trim_result.details:
        print(f"  {detail}")
    print(
        f"Trim result: {trim_result.episodes_trimmed} episode(s) trimmed, "
        f"{trim_result.frames_removed} frame(s) removed, "
        f"{trim_result.episodes_removed} episode(s) dropped"
    )
    if any(detail.startswith("Error:") for detail in trim_result.details):
        return 1

    if args.subsample_fps is not None:
        print(f"\n[4/{total_steps}] Subsampling dataset to {args.subsample_fps}fps...")
        subsample_result = subsample_dataset(
            local_dir,
            target_fps=args.subsample_fps,
            progress=lambda msg: print(f"  {msg}"),
        )
        for detail in subsample_result.details:
            print(f"  {detail}")
        print(
            f"Subsample result: {subsample_result.source_fps}fps -> "
            f"{subsample_result.target_fps}fps, "
            f"{subsample_result.frames_removed} frame(s) removed"
        )
        if any(detail.startswith("Error:") for detail in subsample_result.details):
            return 1

    print(f"\n[{total_steps - 1}/{total_steps}] Running health checks...")
    dataset = load_local(local_dir)
    report = run_checks(dataset, checks=CHECKS)
    print_report(report)
    if report.overall_severity == Severity.FAIL:
        print("Health checks have FAIL results; upload skipped.", file=sys.stderr)
        return 1

    if args.skip_upload:
        print(f"\n[{total_steps}/{total_steps}] Upload skipped by --skip-upload")
        return 0

    print(f"\n[{total_steps}/{total_steps}] Uploading processed dataset...")
    api = HfApi()
    api.create_repo(target_repo, repo_type="dataset", private=args.private, exist_ok=True)
    upload_kwargs = {
        "repo_id": target_repo,
        "repo_type": "dataset",
        "folder_path": str(local_dir),
        "commit_message": f"Upload processed dataset from {source_repo}",
    }
    if args.upload_delete_existing:
        upload_kwargs["delete_patterns"] = "*"
    api.upload_folder(**upload_kwargs)
    print(f"Uploaded: https://huggingface.co/datasets/{target_repo}")
    return 0


def _trimmed_repo_id(repo_id: str, subsample_fps: int | None = None) -> str:
    if "/" not in repo_id:
        raise SystemExit("repo_id must be in namespace/name form")
    namespace, name = repo_id.split("/", 1)
    suffix = "-trimmed"
    if subsample_fps is not None:
        if subsample_fps <= 0:
            raise SystemExit("--subsample-fps must be positive")
        suffix += f"-subsampled-{subsample_fps}fps"
    return f"{namespace}/{name}{suffix}"


if __name__ == "__main__":
    raise SystemExit(main())
