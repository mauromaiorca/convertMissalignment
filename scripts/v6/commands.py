from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CommandPlan:
    stage_id: str
    executable: str
    arguments: list[str] = field(default_factory=list)
    working_directory: str = "."
    environment: dict[str, str] = field(default_factory=dict)
    description: str = ""
    requires_syntax_probe: bool = False

    def argv(self) -> list[str]:
        return [self.executable, *self.arguments]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CommandResult:
    command: CommandPlan
    absolute_executable: str | None
    stdout: str
    stderr: str
    exit_code: int
    start_time: float
    end_time: float
    hostname: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["command"] = self.command.to_dict()
        return data


class CommandRunner:
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def run(self, plan: CommandPlan, *, check: bool = True) -> CommandResult:
        start = time.time()
        env = os.environ.copy()
        env.update(plan.environment)
        exe = shutil.which(plan.executable) or (plan.executable if Path(plan.executable).exists() else None)
        cp = subprocess.run(
            plan.argv(), cwd=plan.working_directory, env=env,
            text=True, capture_output=True, check=False,
        )
        end = time.time()
        result = CommandResult(plan, exe, cp.stdout, cp.stderr, cp.returncode,
                               start, end, socket.getfqdn() or socket.gethostname())
        path = self.log_dir / f"{plan.stage_id}.command.json"
        path.write_text(json.dumps(result.to_dict(), indent=2) + "\n")
        if check and cp.returncode != 0:
            raise RuntimeError(f"command failed for {plan.stage_id}: {cp.returncode}")
        return result

