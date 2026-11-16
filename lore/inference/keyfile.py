"""Where each provider's key file lives, and how a key file is read safely.

Every provider key file is one line `NAME=...` in `~/.config/lore/`, outside
the repository so it can never be committed:

    gemini.env  anthropic.env  openai.env  llm.env  openrouter.env

`config_dir()` is `Path.home() / ".config" / "lore"` except when `HOME` is
set, which it is under Git Bash, WSL, and every test that relocates a home
directory: on POSIX `Path.home()` already honours `HOME`, and on Windows it
reads `USERPROFILE`, so honouring `HOME` here keeps one meaning everywhere.

`mode_is_private()` is the "only its owner can read it" check. POSIX file
modes are real there; Windows has ACLs instead, and `os.stat` reports a
synthetic mode that cannot tell a private file from a shared one, so the
check is skipped there rather than refusing every key file a Windows user
writes. A Windows key file under the user's profile is owner-only by the
profile's own ACL.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path


def config_dir() -> Path:
    home = os.environ.get("HOME", "").strip()
    base = Path(home) if home else Path.home()
    return base / ".config" / "lore"


def key_file(name: str) -> Path:
    """`<config dir>/<name>.env`."""
    return config_dir() / f"{name}.env"


def mode_is_private(path: Path) -> bool:
    """Whether nothing but the owner can read `path` (always True on Windows)."""
    if os.name == "nt":
        return True
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        return False
    return not (mode & ~0o600)
