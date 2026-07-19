"""Frozen robot-model profile adapter with hash verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .paths import resolve_inside_root

VALID_PROFILES = ("g1",)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ModelProfile:
    """One immutable model profile with verified content hashes."""

    profile_id: str
    urdf_path: Path
    urdf_sha256: str
    mjcf_path: Path
    mjcf_sha256: str
    scene_mjcf_path: Path | None
    srdf_path: Path | None
    contact_frames_path: Path | None
    contact_frames_sha256: str | None

    @property
    def cache_key(self) -> str:
        contact = (self.contact_frames_sha256 or "none")[:16]
        return (
            f"{self.profile_id}:{self.urdf_sha256[:16]}:"
            f"{self.mjcf_sha256[:16]}:{contact}"
        )


def load_model_profile(profile_id: str, *, verify_hash: bool = True) -> ModelProfile:
    if profile_id not in VALID_PROFILES:
        raise ValueError(
            f"unknown model profile: {profile_id!r}; valid: {VALID_PROFILES}"
        )
    manifest = json.loads(
        resolve_inside_root("models/MODEL_MANIFEST.json").read_text()
    )
    profile = manifest["profiles"][profile_id]

    def entry(kind: str):
        if kind not in profile:
            return None, None
        descriptor = profile[kind]
        relative = descriptor.get("root_path")
        if relative is None:
            relative = Path("models") / descriptor["path"]
        path = resolve_inside_root(relative)
        expected = profile[kind]["sha256"]
        if verify_hash:
            actual = _sha256(path)
            if actual != expected:
                raise RuntimeError(
                    f"{profile_id} {kind} hash mismatch: {actual} != {expected}"
                )
        return path, expected

    urdf_path, urdf_sha = entry("urdf")
    mjcf_path, mjcf_sha = entry("mjcf")
    scene_path, _ = entry("scene_mjcf")
    contact_frames_path, contact_frames_sha = entry("contact_frames")
    return ModelProfile(
        profile_id=profile_id,
        urdf_path=urdf_path,
        urdf_sha256=urdf_sha,
        mjcf_path=mjcf_path,
        mjcf_sha256=mjcf_sha,
        scene_mjcf_path=scene_path,
        srdf_path=None,
        contact_frames_path=contact_frames_path,
        contact_frames_sha256=contact_frames_sha,
    )
