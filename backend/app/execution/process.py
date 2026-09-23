"""可被 Run 取消的本地子进程运行器。"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Sequence
from threading import Event


class ProcessCancelled(Exception):
    """子进程因所属 Run 被取消而停止。"""


def run_process(
    arguments: Sequence[str],
    *,
    cwd,
    timeout_seconds: int,
    cancel_event: Event | None = None,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(arguments, cwd=cwd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    started = time.monotonic()
    while True:
        if cancel_event and cancel_event.is_set():
            process.kill()
            process.communicate()
            raise ProcessCancelled()
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            process.kill()
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(arguments, timeout_seconds, output=stdout, stderr=stderr)
        try:
            stdout, stderr = process.communicate(timeout=min(0.05, remaining))
        except subprocess.TimeoutExpired:
            continue
        return subprocess.CompletedProcess(arguments, process.returncode, stdout, stderr)
