from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION, SOFTWARE_VERSION
from .config import CapabilitySet


SNAPSHOT_TYPES = (
    "base",
    "alignment_initial",
    "pre_missalign",
    "missalign_smoke",
    "missalign_full",
    "post_missalign",
    "selection",
)


@dataclass
class WarpProjectRef:
    project_id: str
    tilt_series_id: str
    frame_series_settings_file: str
    frame_series_raw_directory: str
    tilt_series_settings_file: str
    tomostar_directory: str
    frame_processing_directory: str
    tilt_series_processing_directory: str
    source_mode: str
    capabilities: CapabilitySet
    geometry_id: str
    ctf_id: str
    selection_id: str
    parent_snapshot_id: str | None
    toml_hash: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["capabilities"] = asdict(self.capabilities)
        return data


@dataclass
class WarpSnapshotRef:
    snapshot_id: str
    snapshot_type: str
    path: str
    parent_snapshot_id: str | None
    created_at: str
    toml_hash: str
    geometry_id: str
    ctf_id: str
    selection_id: str
    copied_files: list[str] = field(default_factory=list)
    linked_files: list[str] = field(default_factory=list)
    mutable_files: list[str] = field(default_factory=list)
    source_hashes: dict[str, Any] = field(default_factory=dict)
    software_versions: dict[str, str] = field(default_factory=dict)
    commands: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class V6Layout:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.manifests = self.output_dir / "manifests"
        self.jobs = self.output_dir / "jobs"
        self.logs = self.output_dir / "logs"
        self.warp = self.output_dir / "warp"
        self.missalignment = self.output_dir / "missalignment"
        self.relion = self.output_dir / "relion"
        self.m = self.output_dir / "m"

    def create(self) -> "V6Layout":
        for sub in (
            "manifests", "jobs", "logs", "logs/stages", "logs/environment", "logs/resources",
            "warp/ingest/frame_series", "warp/ingest/frame_processing", "warp/ingest/tomostar",
            "warp/base/processing", "warp/alignment_initial/processing",
            "warp/pre_missalign/processing", "warp/missalign_smoke/processing",
            "warp/missalign_full/processing", "warp/post_missalign/processing",
            "warp/selections/default", "missalignment/smoke", "missalignment/full",
            "relion", "m",
        ):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)
        return self

    def snapshot_dir(self, snapshot_type: str) -> Path:
        if snapshot_type == "selection":
            return self.warp / "selections" / "default"
        return self.warp / snapshot_type


class SnapshotManager:
    def __init__(self, layout: V6Layout, toml_hash: str):
        self.layout = layout
        self.toml_hash = toml_hash

    def declare_snapshots(self, snapshot_types: list[str]) -> None:
        for snapshot_type in snapshot_types:
            root = self.layout.snapshot_dir(snapshot_type)
            root.mkdir(parents=True, exist_ok=True)
            manifest = root / "snapshot_manifest.json"
            if not manifest.exists():
                manifest.write_text(json.dumps({
                    "snapshot_type": snapshot_type,
                    "status": "declared",
                    "materialised": False,
                    "validated": False,
                    "toml_hash": self.toml_hash,
                }, indent=2) + "\n")

    def create_snapshot(
        self,
        snapshot_type: str,
        *,
        parent_snapshot_id: str | None,
        geometry_id: str = "geometry_initial",
        ctf_id: str = "ctf_unset",
        selection_id: str = "selection_unset",
        copy_files: list[Path] | None = None,
        link_files: list[Path] | None = None,
        mutable_files: list[Path] | None = None,
    ) -> WarpSnapshotRef:
        if snapshot_type not in SNAPSHOT_TYPES:
            raise ValueError(f"unknown snapshot type {snapshot_type!r}")
        root = self.layout.snapshot_dir(snapshot_type)
        root.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        linked: list[str] = []
        mutable: list[str] = []
        for src in copy_files or []:
            dst = root / Path(src).name
            if Path(src).resolve() != dst.resolve():
                shutil.copy2(src, dst)
            copied.append(str(dst))
        for src in link_files or []:
            dst = root / Path(src).name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(Path(src).resolve())
            linked.append(str(dst))
        for src in mutable_files or []:
            dst = root / Path(src).name
            if Path(src).resolve() != dst.resolve():
                shutil.copy2(src, dst)
            mutable.append(str(dst))
        snap = WarpSnapshotRef(
            snapshot_id=f"{snapshot_type}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}_{self.toml_hash[:8]}",
            snapshot_type=snapshot_type,
            path=str(root),
            parent_snapshot_id=parent_snapshot_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            toml_hash=self.toml_hash,
            geometry_id=geometry_id,
            ctf_id=ctf_id,
            selection_id=selection_id,
            copied_files=copied,
            linked_files=linked,
            mutable_files=mutable,
            software_versions={"working_scripts_v6": SOFTWARE_VERSION},
        )
        (root / "snapshot_manifest.json").write_text(json.dumps(snap.to_dict(), indent=2) + "\n")
        return snap


def write_project_ref(layout: V6Layout, ref: WarpProjectRef) -> Path:
    path = layout.manifests / "warp_project_ref.json"
    path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "software_version": SOFTWARE_VERSION,
        "warp_project": ref.to_dict(),
    }, indent=2) + "\n")
    return path
