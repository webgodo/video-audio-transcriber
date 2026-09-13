"""Browser front end for the transcriber (optional extra)."""

from __future__ import annotations

import sys

HINT = (
    "the web interface needs extra packages:\n"
    "    pip install 'video-audio-transcriber[web]'\n"
    "(it requires Python 3.10 or newer)"
)


def main() -> int:
    """Entry point for the ``vatfa-web`` console script."""
    try:
        from .app import main as run
    except ImportError as exc:
        print(f"error: {exc}\n{HINT}", file=sys.stderr)
        return 1
    return run()
