"""Shared, stdlib-only helpers for standalone entry-point scripts (run.py,
sync_data.py). Pure stdlib deliberately - these run *before* requirements.txt
is guaranteed to be installed, so nothing here may import a third-party
package at module scope.
"""

from __future__ import annotations

import hashlib
import os
import platform
import socket
import subprocess
import sys
import venv
from pathlib import Path

VENV_MARKER = "TENNIS_APP_IN_VENV"


def _venv_python(venv_dir: Path) -> Path:
    if platform.system() == "Windows":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def bootstrap_venv(entry_script: Path) -> None:
    """No-op if already running inside the managed venv; otherwise create it
    (if missing), install requirements.txt (if changed), and re-exec `entry_script`
    inside it.

    `entry_script` must be the file currently being run (e.g. `Path(__file__)`
    from the caller) - re-exec always restarts *that* script, not run.py, so
    this is safe to call from any launcher-style entry point in the project.
    """
    if os.environ.get(VENV_MARKER) == "1":
        return

    root = entry_script.resolve().parent
    venv_dir = root / ".venv"
    requirements = root / "requirements.txt"

    python_path = _venv_python(venv_dir)
    freshly_created = False
    if not python_path.exists():
        print("Creating an isolated virtual environment in .venv/ ...")
        venv.EnvBuilder(with_pip=True).create(venv_dir)
        freshly_created = True

    stamp = venv_dir / ".requirements.sha256"
    req_hash = hashlib.sha256(requirements.read_bytes()).hexdigest()
    needs_install = freshly_created or not stamp.exists() or stamp.read_text().strip() != req_hash

    if needs_install:
        print("Installing dependencies into .venv/ (only happens when requirements.txt changes)...")
        subprocess.run([str(python_path), "-m", "pip", "install", "--quiet", "--upgrade", "pip"], check=True)
        subprocess.run([str(python_path), "-m", "pip", "install", "--quiet", "-r", str(requirements)], check=True)
        stamp.write_text(req_hash)

    env = os.environ.copy()
    env[VENV_MARKER] = "1"
    os.execve(str(python_path), [str(python_path), str(entry_script.resolve()), *sys.argv[1:]], env)


def get_lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no packet is actually sent (UDP), just used to pick a route
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()
