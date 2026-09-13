"""Run the Persian clean-up over text and subtitle files, with no model.

``normalize.py`` fixes Whisper's Persian output. The same rules are just as
useful on text that never came from this tool: YouTube auto-captions, a
subtitle file downloaded from the web, or something a person typed. This
module applies them to ``.txt``, ``.srt`` and ``.vtt`` files, which means the
most distinctive part of the project runs with no model download, no GPU and
no media file.

The subtitle handling is deliberately not a parser. It locates the timing line
in each block and rewrites only what follows it, passing everything else
through byte for byte. A malformed file therefore comes out no more broken
than it went in, and timings can never be corrupted.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import IO, List, Optional, Union

from .normalize import normalize

log = logging.getLogger("video_audio_transcriber")

ARROW = "-->"
KINDS = ("txt", "srt", "vtt")
#: Blocks opening with one of these are WebVTT metadata, never cue text.
_VTT_KEYWORDS = ("WEBVTT", "NOTE", "STYLE", "REGION")
#: Tried in order. cp1256 (Windows Arabic) is common in Persian subtitle files
#: found online and decodes any byte, so it must come last.
_ENCODINGS = ("utf-8-sig", "cp1256")


def detect_kind(text: str, name: str = "") -> str:
    """Guess whether ``text`` is plain text, SRT or WebVTT.

    The file name wins when it carries a known suffix, since that is what the
    author meant; otherwise fall back to sniffing the content.
    """
    suffix = Path(name).suffix.lower()
    if suffix in (".srt", ".vtt"):
        return suffix[1:]
    head = text.lstrip("﻿").lstrip()
    if head.upper().startswith("WEBVTT"):
        return "vtt"
    if ARROW in text[:4000]:
        return "srt"
    return "txt"


def read_text(path: Union[str, Path]) -> str:
    """Read a text file, tolerating the encodings Persian subtitles come in."""
    raw = Path(path).read_bytes()
    for encoding in _ENCODINGS:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if encoding != _ENCODINGS[0]:
            log.info("%s: decoded as %s", path, encoding)
        return text
    # cp1256 maps every byte, so this is unreachable in practice.
    return raw.decode("utf-8", errors="replace")


def _line_ending(text: str) -> str:
    """Keep the file's own line ending instead of imposing one."""
    return "\r\n" if "\r\n" in text else "\n"


def _normalize_line(line: str, **opts: object) -> str:
    # Blank and whitespace-only lines carry structure; leave them exactly.
    return normalize(line, **opts) if line.strip() else line


def fix_txt(text: str, **opts: object) -> str:
    """Clean up plain text, preserving blank lines and paragraph structure."""
    ending = _line_ending(text)
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return ending.join(_normalize_line(line, **opts) for line in lines)


def _fix_block(lines: List[str], vtt: bool, **opts: object) -> List[str]:
    if vtt and lines and lines[0].lstrip("﻿").upper().startswith(_VTT_KEYWORDS):
        return lines  # metadata block: verbatim
    timing = next((i for i, line in enumerate(lines) if ARROW in line), None)
    if timing is None:
        return lines  # no timing line, so nothing here is provably cue text
    # Cue number and identifier lines sit above the timing line; text below it.
    return lines[: timing + 1] + [_normalize_line(line, **opts) for line in lines[timing + 1 :]]


def _fix_cues(text: str, vtt: bool, **opts: object) -> str:
    ending = _line_ending(text)
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: List[str] = []
    block: List[str] = []
    for line in lines:
        if line.strip():
            block.append(line)
            continue
        out.extend(_fix_block(block, vtt, **opts))
        out.append(line)
        block = []
    out.extend(_fix_block(block, vtt, **opts))
    return ending.join(out)


def fix_srt(text: str, **opts: object) -> str:
    """Clean up SRT cue text, leaving indices and timestamps untouched."""
    return _fix_cues(text, vtt=False, **opts)


def fix_vtt(text: str, **opts: object) -> str:
    """Clean up WebVTT cue text, leaving the header, metadata and timings untouched."""
    return _fix_cues(text, vtt=True, **opts)


FIXERS = {"txt": fix_txt, "srt": fix_srt, "vtt": fix_vtt}


def fix_text(text: str, *, kind: str = "auto", name: str = "", **opts: object) -> str:
    """Clean up ``text``, detecting the format unless ``kind`` says otherwise."""
    if kind == "auto":
        kind = detect_kind(text, name)
    if kind not in FIXERS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
    return FIXERS[kind](text, **opts)


def fix_stream(
    src: IO[str],
    dst: IO[str],
    *,
    kind: str = "auto",
    name: str = "",
    **opts: object,
) -> None:
    """Clean up everything readable from ``src`` and write it to ``dst``."""
    dst.write(fix_text(src.read(), kind=kind, name=name, **opts))


def fix_file(
    path: Path,
    out_dir: Optional[Path] = None,
    stdout: Optional[IO[str]] = None,
    **opts: object,
) -> Optional[Path]:
    """Clean up one file, to ``out_dir`` or to ``stdout``.

    Returns the path written, or ``None`` when the result went to the stream.
    Writing back over the input is refused: for many people the subtitle file
    they are cleaning up is the only copy they have.
    """
    fixed = fix_text(read_text(path), name=path.name, **opts)
    if out_dir is None:
        (stdout if stdout is not None else sys.stdout).write(fixed)
        return None
    target = out_dir / path.name
    if target.resolve() == path.resolve():
        raise ValueError(f"refusing to overwrite the input file: {path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(fixed, encoding="utf-8", newline="")
    return target
