from __future__ import annotations

from pathlib import Path

from .config import ProjectConfig


def run(cfg: ProjectConfig, *, settings: Path, run_dir: Path, toml_hash: str) -> None:
    raise RuntimeError(
        "20_initial_alignment_and_qc is blocked: the real v5 Warp XML conversion backend "
        "has not been wired into v6 yet; refusing to append private XML metadata"
    )
