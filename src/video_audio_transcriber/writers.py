"""Output writers (txt, srt, vtt, json, tsv, html) and subtitle cue splitting."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import IO, Callable, Dict, Iterable, List

from .transcriber import Segment, Transcript, Word

RLM = "\u200f"  # RIGHT-TO-LEFT MARK


def format_timestamp(seconds: float, decimal: str = ",") -> str:
    """``HH:MM:SS,mmm`` (SRT) or ``HH:MM:SS.mmm`` (VTT)."""
    total_ms = max(0, int(round(seconds * 1000)))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, ms = divmod(total_ms, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal}{ms:03d}"


def _cue_text(segment: Segment, rtl_mark: bool) -> str:
    text = " ".join(segment.text.split())
    if text and rtl_mark:
        text = RLM + text
    return text


def _cue_end(segment: Segment) -> float:
    # Players ignore zero-length cues.
    return max(segment.end, segment.start + 0.05)


def write_txt(transcript: Transcript, fh: IO[str], **_: object) -> None:
    for segment in transcript.segments:
        text = segment.text.strip()
        if text:
            fh.write(text + "\n")


def write_srt(transcript: Transcript, fh: IO[str], rtl_mark: bool = False, **_: object) -> None:
    index = 0
    for segment in transcript.segments:
        text = _cue_text(segment, rtl_mark)
        if not text:
            continue
        index += 1
        fh.write(
            f"{index}\n"
            f"{format_timestamp(segment.start)} --> {format_timestamp(_cue_end(segment))}\n"
            f"{text}\n\n"
        )


def write_vtt(transcript: Transcript, fh: IO[str], rtl_mark: bool = False, **_: object) -> None:
    fh.write("WEBVTT\n\n")
    for segment in transcript.segments:
        text = _cue_text(segment, rtl_mark)
        if not text:
            continue
        fh.write(
            f"{format_timestamp(segment.start, '.')} --> {format_timestamp(_cue_end(segment), '.')}\n"
            f"{text}\n\n"
        )


def to_dict(transcript: Transcript) -> dict:
    segments = []
    for s in transcript.segments:
        item = {
            "id": s.id,
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "text": s.text,
            "avg_logprob": round(s.avg_logprob, 4),
            "no_speech_prob": round(s.no_speech_prob, 4),
        }
        if s.words:
            item["words"] = [
                {
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "word": w.word.strip(),
                    "probability": round(w.probability, 4),
                }
                for w in s.words
            ]
        segments.append(item)
    return {
        # File name only. These outputs get shared, and an absolute path leaks
        # the directory layout (and often the client name) of whoever ran it.
        "source": Path(transcript.source).name,
        "model": transcript.model,
        "language": transcript.language,
        "language_probability": round(transcript.language_probability, 4),
        "duration": round(transcript.duration, 3),
        "text": transcript.text,
        "segments": segments,
    }


def write_json(transcript: Transcript, fh: IO[str], **_: object) -> None:
    json.dump(to_dict(transcript), fh, ensure_ascii=False, indent=2)
    fh.write("\n")


def write_tsv(transcript: Transcript, fh: IO[str], **_: object) -> None:
    fh.write("start\tend\ttext\n")
    for segment in transcript.segments:
        text = " ".join(segment.text.split())
        if text:
            fh.write(f"{int(round(segment.start * 1000))}\t{int(round(segment.end * 1000))}\t{text}\n")


def write_html(
    transcript: Transcript,
    fh: IO[str],
    media: object = None,
    embed_media: bool = False,
    **_: object,
) -> None:
    """Write a self-contained interactive page (see :mod:`html_view`).

    The import is deferred so the rest of this module stays free of it, the
    same way the model and progress-bar imports are deferred elsewhere.
    """
    from .html_view import render_html

    # Writers only receive the handle, but the page needs to know where it is
    # being written so it can link the media relatively. StringIO has no name.
    dest = getattr(fh, "name", None)
    fh.write(
        render_html(
            transcript,
            media=media if isinstance(media, (str, Path)) else None,
            dest=dest if isinstance(dest, str) else None,
            embed_media=bool(embed_media),
        )
    )


WRITERS: Dict[str, Callable[..., None]] = {
    "txt": write_txt,
    "srt": write_srt,
    "vtt": write_vtt,
    "json": write_json,
    "tsv": write_tsv,
    "html": write_html,
}

#: Every supported ``-f`` value, in the order they are offered on the CLI.
#: Derived from :data:`WRITERS` (dicts keep insertion order) so the two can
#: never drift apart; registering a writer is all it takes to add a format.
FORMATS = tuple(WRITERS)

#: Formats made of timed cues, which are written from the subtitle-split copy
#: of the transcript rather than from Whisper's own long segments.
CUE_FORMATS = frozenset({"srt", "vtt"})


def write_transcript(transcript: Transcript, fmt: str, path, **options: object) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        WRITERS[fmt](transcript, fh, **options)


# ------------------------------------------------------------- cue splitting


def _make_cue(words: List[Word], template: Segment) -> Segment:
    return Segment(
        id=0,
        start=words[0].start,
        end=words[-1].end,
        text=" ".join(w.word.strip() for w in words),
        words=list(words),
        avg_logprob=template.avg_logprob,
        no_speech_prob=template.no_speech_prob,
    )


def split_for_subtitles(
    segments: Iterable[Segment],
    max_chars: int = 42,
    max_duration: float = 7.0,
    max_gap: float = 1.0,
) -> List[Segment]:
    """Re-chunk segments into subtitle-sized cues using word timestamps.

    Whisper segments can be a full 30-second window; players show them as a
    wall of text. This splits on the character budget, a maximum duration
    and pauses longer than ``max_gap`` seconds. Segments that have no word
    timestamps (or already fit) are passed through unchanged.
    """
    out: List[Segment] = []
    for segment in segments:
        words = [w for w in segment.words if w.word.strip()]
        fits = len(segment.text) <= max_chars and (segment.end - segment.start) <= max_duration
        if not words or fits:
            out.append(segment)
            continue
        chunk: List[Word] = []
        chunk_len = 0
        for word in words:
            wlen = len(word.word.strip())
            if chunk:
                too_long = chunk_len + 1 + wlen > max_chars
                too_slow = word.end - chunk[0].start > max_duration
                big_gap = word.start - chunk[-1].end > max_gap
                if too_long or too_slow or big_gap:
                    out.append(_make_cue(chunk, segment))
                    chunk, chunk_len = [], 0
            chunk.append(word)
            chunk_len += wlen + (1 if chunk_len else 0)
        if chunk:
            out.append(_make_cue(chunk, segment))
    return [replace(s, id=i) for i, s in enumerate(out)]
