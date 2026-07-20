from __future__ import annotations

import json
import shutil
import subprocess
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from .commands import CommandPlan


@dataclass
class WarpToolsSyntax:
    executable: str
    executable_path: str = ""
    version: str = "unavailable"
    help_text: str = ""
    command_help: dict[str, str] = field(default_factory=dict)
    help_hashes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "executable": self.executable,
            "executable_path": self.executable_path,
            "version": self.version,
            "help_text": self.help_text,
            "command_help": self.command_help,
            "help_hashes": self.help_hashes,
        }


class WarpToolsAdapter:
    STACK_INGEST_COMMAND = "stack-ingest"
    VALIDATE_PROJECT_COMMAND = "validate-project"
    STACK_REQUIRED_FLAGS = (
        "--input-stack",
        "--tilt-file",
        "--output-dir",
        "--series-id",
        "--pixel-size",
    )

    def __init__(self, executable: str = "WarpTools"):
        self.executable = executable

    def _resolved_executable(self) -> str | None:
        found = shutil.which(self.executable)
        if found:
            return str(Path(found).resolve())
        p = Path(self.executable)
        if p.exists():
            return str(p.resolve())
        return None

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def probe(
        self,
        cache_path: Path | None = None,
        commands: tuple[str, ...] = (STACK_INGEST_COMMAND, VALIDATE_PROJECT_COMMAND),
    ) -> WarpToolsSyntax:
        syntax = WarpToolsSyntax(executable=self.executable)
        resolved = self._resolved_executable()
        if resolved is None:
            if cache_path:
                cache_path.write_text(json.dumps(syntax.to_dict(), indent=2) + "\n")
            return syntax
        syntax.executable_path = resolved
        version = subprocess.run([self.executable, "--version"], text=True, capture_output=True, check=False)
        syntax.version = (version.stdout or version.stderr).strip() or "version unavailable"
        help_cp = subprocess.run([self.executable, "--help"], text=True, capture_output=True, check=False)
        syntax.help_text = (help_cp.stdout or help_cp.stderr)
        syntax.help_hashes["__root__"] = self._hash(syntax.help_text)
        for command in commands:
            cp = subprocess.run([self.executable, command, "--help"], text=True, capture_output=True, check=False)
            text = cp.stdout or cp.stderr
            syntax.command_help[command] = text
            syntax.help_hashes[command] = self._hash(text)
        if cache_path:
            cache_path.write_text(json.dumps(syntax.to_dict(), indent=2) + "\n")
        return syntax

    def supports_stack_only_ingest(self, cache_path: Path | None = None) -> tuple[bool, str, WarpToolsSyntax]:
        syntax = self.probe(cache_path)
        if not syntax.executable_path:
            return False, f"WarpTools executable unresolved: {self.executable}", syntax
        ingest_help = syntax.command_help.get(self.STACK_INGEST_COMMAND, "")
        validate_help = syntax.command_help.get(self.VALIDATE_PROJECT_COMMAND, "")
        missing = [flag for flag in self.STACK_REQUIRED_FLAGS if flag not in ingest_help]
        if missing:
            return False, (
                f"WarpTools stack-only command contract unresolved: {self.STACK_INGEST_COMMAND} "
                f"help is missing {missing}"
            ), syntax
        if "--project-dir" not in validate_help:
            return False, (
                f"WarpTools project validation contract unresolved: {self.VALIDATE_PROJECT_COMMAND} "
                "help is missing --project-dir"
            ), syntax
        return True, "", syntax

    def run_stack_only_ingest(
        self,
        *,
        input_stack: Path,
        tilt_file: Path,
        output_dir: Path,
        series_id: str,
        pixel_size_A: float,
        env: dict[str, str] | None = None,
    ) -> dict:
        ok, reason, syntax = self.supports_stack_only_ingest()
        if not ok:
            raise RuntimeError(reason)
        cmd = [
            self.executable,
            self.STACK_INGEST_COMMAND,
            "--input-stack", str(input_stack),
            "--tilt-file", str(tilt_file),
            "--output-dir", str(output_dir),
            "--series-id", series_id,
            "--pixel-size", f"{pixel_size_A:.12g}",
        ]
        cp = subprocess.run(cmd, text=True, capture_output=True, check=False, env=env)
        if cp.returncode != 0:
            raise RuntimeError(
                f"WarpTools stack ingest failed rc={cp.returncode}: {' '.join(cmd)}\n{cp.stderr or cp.stdout}"
            )
        validate_cmd = [
            self.executable,
            self.VALIDATE_PROJECT_COMMAND,
            "--project-dir", str(output_dir),
        ]
        vp = subprocess.run(validate_cmd, text=True, capture_output=True, check=False, env=env)
        if vp.returncode != 0:
            raise RuntimeError(
                f"WarpTools project validation failed rc={vp.returncode}: {' '.join(validate_cmd)}\n"
                f"{vp.stderr or vp.stdout}"
            )
        return {
            "syntax": syntax.to_dict(),
            "commands": [
                {"argv": cmd, "stdout": cp.stdout, "stderr": cp.stderr, "returncode": cp.returncode},
                {"argv": validate_cmd, "stdout": vp.stdout, "stderr": vp.stderr, "returncode": vp.returncode},
            ],
        }

    def stack_only_ingest_plan(self, toml: Path, project_dir: Path) -> list[CommandPlan]:
        return [
            CommandPlan("10_warp_ingest", self.executable, [],
                        str(project_dir),
                        description="stack-only WarpTools ingest requires syntax probe before execution",
                        requires_syntax_probe=True)
        ]

    def movie_ingest_plan(self, toml: Path, project_dir: Path) -> list[CommandPlan]:
        return [
            CommandPlan("10_warp_ingest", self.executable, [],
                        str(project_dir),
                        description="movie WarpTools ingest requires syntax probe before execution",
                        requires_syntax_probe=True)
        ]

    def post_missalign_plan(self, toml: Path, project_dir: Path) -> list[CommandPlan]:
        return [
            CommandPlan("40_warp_postprocess", self.executable, [],
                        str(project_dir),
                        description="post-MissAlignment CTF/reconstruction/QC requires syntax probe before execution",
                        requires_syntax_probe=True)
        ]
