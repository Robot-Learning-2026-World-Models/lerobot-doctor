"""Tests for the Hugging Face trim workflow script."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import trim_hf_dataset  # noqa: E402


def test_trimmed_repo_id_appends_suffix():
    assert (
        trim_hf_dataset._trimmed_repo_id("rl26-world-models/so101-task1-720p-whole-arm")
        == "rl26-world-models/so101-task1-720p-whole-arm-trimmed"
    )


def test_trimmed_repo_id_requires_namespace():
    with pytest.raises(SystemExit):
        trim_hf_dataset._trimmed_repo_id("so101-task1-720p-whole-arm")

