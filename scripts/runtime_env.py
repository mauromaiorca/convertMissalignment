#!/usr/bin/env python3
"""Small, standard-library-only Python environment bootstrap for CLI entry points."""
from __future__ import annotations

import importlib.util
import os
import sys
import tomllib
from pathlib import Path
from typing import Iterable

_REEXEC_GUARD = "MISSALIGN_PYTHON_REEXEC"


def _configured_environment(settings: Path | None) -> str | None:
    if not settings or not settings.is_file():
        return None
    try:
        with settings.open("rb") as fh:
            cfg = tomllib.load(fh)
    except Exception:
        return None
    value = (cfg.get("cluster", {}) or {}).get("environment")
    return str(value).strip() if value else None


def _environment_candidate(
    *, settings: Path | None, explicit_env: str | None
) -> str | None:
    """Resolve a scientific environment without assuming a user-specific path."""
    candidates = (
        explicit_env,
        os.environ.get("MISSALIGN_ENV"),
        _configured_environment(settings),
    )
    for candidate in candidates:
        value = str(candidate or "").strip()
        if value:
            return value
    return None


def missing_modules(required: Iterable[str]) -> list[str]:
    return [name for name in required if importlib.util.find_spec(name) is None]


def ensure_scientific_python(
    *,
    script: Path,
    argv: list[str],
    settings: Path | None = None,
    explicit_env: str | None = None,
    required: tuple[str, ...] = ("numpy", "mrcfile"),
    label: str = "command",
) -> None:
    """Re-exec under a configured environment when required modules are absent.

    The current interpreter is used when it already provides the requested modules.
    Otherwise, the environment must be supplied with ``--missalign-env``, the
    ``MISSALIGN_ENV`` variable, or ``[cluster].environment`` in project settings.
    """
    state = "after re-exec" if os.environ.get(_REEXEC_GUARD) == "1" else "active"
    print(f"[env] {state} Python: {sys.executable}", flush=True)

    missing = missing_modules(required)
    if not missing:
        return

    env_dir = _environment_candidate(settings=settings, explicit_env=explicit_env)
    if not env_dir:
        raise SystemExit(
            f"ERROR: {label} requires {', '.join(required)}, but the current interpreter "
            f"({sys.executable}) lacks {', '.join(missing)}. Activate a suitable environment, "
            "pass --missalign-env, set MISSALIGN_ENV, or configure [cluster].environment."
        )

    env_python = Path(env_dir).expanduser() / "bin" / "python"
    try:
        same_python = env_python.resolve() == Path(sys.executable).resolve()
    except OSError:
        same_python = False

    if same_python or os.environ.get(_REEXEC_GUARD) == "1":
        raise SystemExit(
            "ERROR: the selected MissAlignment Python still lacks required modules: "
            + ", ".join(missing)
            + f". Interpreter: {sys.executable}. Environment: {env_dir}."
        )

    if not env_python.is_file() or not os.access(env_python, os.X_OK):
        raise SystemExit(
            f"ERROR: no executable Python was found at {env_python}. Activate the "
            "environment or correct --missalign-env / MISSALIGN_ENV / [cluster].environment."
        )

    print(f"[env] re-exec under {env_python} (missing here: {', '.join(missing)})", flush=True)
    env = dict(os.environ)
    env[_REEXEC_GUARD] = "1"
    os.execve(str(env_python), [str(env_python), str(script.resolve()), *argv], env)
