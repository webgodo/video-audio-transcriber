"""Fetch media from a URL so it can be transcribed directly.

``vatfa https://...`` downloads the audio with yt-dlp and hands the file to
the ordinary pipeline. This is the one part of the tool that touches the
network; transcription itself still happens entirely on your machine.

yt-dlp is used as a library rather than a subprocess, so there is no PATH
problem and failures arrive as real exceptions. It is also deliberately asked
for ``bestaudio`` with no post-processing: decoding goes through PyAV, which
bundles its own FFmpeg, so this project needs no ``ffmpeg`` binary installed.
Adding an extract-audio post-processor would quietly make one a requirement.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, List, Optional, Sequence

log = logging.getLogger("video_audio_transcriber")

#: A scheme followed by "://". Requiring the slashes keeps Windows drive
#: letters ("C:\media\a.mp4") from looking like URLs.
URL_RE = re.compile(r"^[a-z][a-z0-9+.\-]*://", re.IGNORECASE)

INSTALL_HINT = (
    "downloading from a URL needs yt-dlp: pip install 'video-audio-transcriber[url]'"
)


def is_url(raw: str) -> bool:
    return bool(URL_RE.match(raw.strip()))


class _Logger:
    """Route yt-dlp's own output into this tool's logger.

    ``quiet`` does not stop yt-dlp writing errors straight to stderr, which
    means a failed download is reported twice: once by yt-dlp and once by us.
    Sending everything to the debug channel keeps the single message from
    ``resolve_urls`` and still shows the full detail under ``-v``.
    """

    def debug(self, message: str) -> None:
        log.debug("yt-dlp: %s", message)

    info = debug
    warning = debug
    error = debug


def _yt_dlp() -> Any:
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(INSTALL_HINT) from exc
    return yt_dlp


def fetch(url: str, dest: Path, *, verbose: bool = False) -> Path:
    """Download the audio at ``url`` into ``dest`` and return the file path."""
    yt_dlp = _yt_dlp()
    dest.mkdir(parents=True, exist_ok=True)
    options = {
        "format": "bestaudio/best",
        "outtmpl": str(dest / "%(title).100B [%(id)s].%(ext)s"),
        "noplaylist": True,  # or one channel URL becomes a 200-video job
        # yt-dlp's own progress bar fights with this tool's, and its
        # post-processor chatter is not something a user asked to see. Our own
        # log lines cover what matters; -v turns the full output back on.
        "quiet": not verbose,
        "no_warnings": not verbose,
        "noprogress": not verbose,
        "retries": 10,
        "fragment_retries": 10,
    }
    if not verbose:
        options["logger"] = _Logger()
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
        if info.get("is_live"):
            raise RuntimeError(f"{url} is a live stream, which never finishes downloading")
        if info.get("_type") == "playlist":  # pragma: no cover - needs the network
            entries = [e for e in (info.get("entries") or []) if e]
            if not entries:
                raise RuntimeError(f"{url} has nothing downloadable in it")
            info = entries[0]
        title = info.get("title") or url
        duration = info.get("duration")
        if duration:
            log.info("downloading %s (%d:%02d)", title, int(duration) // 60, int(duration) % 60)
        else:
            log.info("downloading %s", title)
        info = ydl.extract_info(url, download=True)
        if info.get("_type") == "playlist":  # pragma: no cover - needs the network
            info = [e for e in (info.get("entries") or []) if e][0]
        path = Path(ydl.prepare_filename(info))
    if not path.is_file():
        # A post-processor or a format merge can change the extension.
        matches = sorted(dest.glob(f"{path.stem}.*"))
        if not matches:  # pragma: no cover - needs the network
            raise RuntimeError(f"downloaded {url} but cannot find the file in {dest}")
        path = matches[0]
    log.info("saved %s", path)
    return path


def resolve_urls(
    raw_inputs: Sequence[str],
    dest: Path,
    *,
    verbose: bool = False,
    on_error: Optional[Any] = None,
) -> List[str]:
    """Replace URLs in ``raw_inputs`` with downloaded paths, leaving paths alone.

    A URL that fails is reported and skipped so the rest of a batch still
    runs, matching how the per-file loop already handles a bad media file.
    """
    resolved: List[str] = []
    for raw in raw_inputs:
        if not is_url(raw):
            resolved.append(raw)
            continue
        try:
            resolved.append(str(fetch(raw, dest, verbose=verbose)))
        except Exception as exc:  # noqa: BLE001 - one bad URL must not stop the batch
            log.error("%s: %s", raw, str(exc).splitlines()[0])
            if on_error is not None:
                on_error(raw, exc)
    return resolved
