#!/usr/bin/env python3
"""Compatibility entry point for TOML-driven IMOD reconstruction and export."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline.imod_reconstruction import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
