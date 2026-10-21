"""Deterministic YAML writing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel

from cosheaf.core.artifact import AlignmentReview, BaseArtifact, VerificationPolicy


def _to_plain_data(data: Any) -> Any:
    if isinstance(data, BaseArtifact):
        dumped = data.model_dump(mode="json")
        if not data.formalizations:
            dumped.pop("formalizations", None)
        if data.alignment == AlignmentReview():
            dumped.pop("alignment", None)
        elif isinstance(dumped.get("alignment"), dict) and (
            dumped["alignment"].get("reviewed_revision") is None
        ):
            dumped["alignment"].pop("reviewed_revision", None)
        if data.verification_policy == VerificationPolicy():
            dumped.pop("verification_policy", None)
        if not data.failure_log:
            dumped.pop("failure_log", None)
        if isinstance(dumped.get("review"), dict) and (
            dumped["review"].get("reviewed_revision") is None
        ):
            dumped["review"].pop("reviewed_revision", None)
        if isinstance(dumped.get("evidence"), list):
            for entry in dumped["evidence"]:
                if isinstance(entry, dict) and entry.get("target_revision") is None:
                    entry.pop("target_revision", None)
        return dumped
    if isinstance(data, BaseModel):
        dumped = data.model_dump(mode="json")
        if isinstance(dumped, dict) and dumped.get("reviewed_revision") is None:
            dumped.pop("reviewed_revision", None)
        return dumped
    if isinstance(data, dict):
        copied = dict(data)
        if "reviewed_revision" in copied and copied["reviewed_revision"] is None:
            copied.pop("reviewed_revision", None)
        return copied
    return data


def dump_yaml_deterministic(data: Any) -> str:
    """Serialize data to YAML without sorting mapping keys."""
    text = yaml.safe_dump(
        _to_plain_data(data),
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    return text if text.endswith("\n") else f"{text}\n"


def write_yaml_deterministic(path: Path, data: Any) -> None:
    """Write deterministic YAML to a path, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_yaml_deterministic(data), encoding="utf-8", newline="\n")
