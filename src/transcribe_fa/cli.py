"""Command-line interface: ``transcribe-fa FILE...``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Sequence

from . import __version__
from .normalize import DIGIT_MODES, normalize, normalize_transcript
from .transcriber import (
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    MODELS,
    Segment,
    audio_duration,
    download_model,
    load_audio,
    load_model,
    resolve_compute_type,
    resolve_device,
    transcribe,
    warm_up,
)
from .writers import FORMATS, split_for_subtitles, write_transcript

log = logging.getLogger("transcribe_fa")

MEDIA_EXTENSIONS = {
    # audio
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".oga", ".opus", ".aac", ".wma",
    ".aiff", ".aif", ".amr", ".mka", ".ac3", ".caf",
    # video
    ".mp4", ".m4v", ".mkv", ".webm", ".mov", ".avi", ".mpg", ".mpeg", ".wmv",
    ".3gp", ".ts", ".mts", ".flv",
}
DEFAULT_FORMATS = ("txt", "srt")

EPILOG = """\
examples:
  transcribe-fa lecture.mp3                       # writes lecture.txt + lecture.srt next to it
  transcribe-fa video.mp4 -f srt --max-cue-chars 42
  transcribe-fa *.m4a -o out/ -f txt,json --word-timestamps
  transcribe-fa interview.wav --stdout | less
  transcribe-fa recordings/ -m medium --device cpu
  transcribe-fa --download-only                   # fetch the default model ahead of time
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcribe-fa",
        description="Offline Persian (Farsi) audio/video transcription with OpenAI Whisper.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("inputs", nargs="*", metavar="FILE", help="audio/video files or directories")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    out = p.add_argument_group("output")
    out.add_argument("-o", "--output-dir", type=Path, metavar="DIR",
                     help="write outputs here (default: next to each input file)")
    out.add_argument("-f", "--format", dest="formats", action="append", metavar="FMT",
                     help="output format(s), comma-separated or repeated: "
                          f"{', '.join(FORMATS)} (default: {','.join(DEFAULT_FORMATS)})")
    out.add_argument("--stdout", action="store_true",
                     help="print the transcript text to stdout (no files unless -f is given)")
    out.add_argument("--max-cue-chars", type=int, default=0, metavar="N",
                     help="split srt/vtt cues longer than N characters using word timestamps "
                          "(0 = keep Whisper's segments)")
    out.add_argument("--rtl-mark", action="store_true",
                     help="prefix subtitle lines with U+200F so trailing punctuation renders "
                          "correctly in players without proper bidi support")
    out.add_argument("--skip-existing", action="store_true",
                     help="skip inputs whose output files already exist")

    m = p.add_argument_group("model")
    m.add_argument("-m", "--model", default=DEFAULT_MODEL,
                   help=f"Whisper model name or path (default: {DEFAULT_MODEL}; see --list-models)")
    m.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                   help="where to run (default: auto = GPU if available)")
    m.add_argument("--compute-type", default="auto", metavar="TYPE",
                   help="int8, int8_float16, float16, float32 ... "
                        "(default: auto = float16 on GPU, int8 on CPU)")
    m.add_argument("--threads", type=int, default=0, metavar="N",
                   help="CPU threads (default: half of the logical cores)")
    m.add_argument("--model-dir", type=Path, metavar="DIR",
                   help="model cache directory (default: the Hugging Face cache)")
    m.add_argument("--offline", action="store_true",
                   help="never download; fail if the model is not cached")
    m.add_argument("--list-models", action="store_true", help="list model sizes and exit")
    m.add_argument("--download-only", action="store_true", help="download the model and exit")

    d = p.add_argument_group("decoding")
    d.add_argument("-l", "--language", default=DEFAULT_LANGUAGE, metavar="LANG",
                   help=f"language code, or 'auto' to detect (default: {DEFAULT_LANGUAGE}; "
                        "forcing it avoids Arabic/Urdu misdetection)")
    d.add_argument("--task", choices=("transcribe", "translate"), default="transcribe",
                   help="'translate' produces English text instead")
    d.add_argument("--beam-size", type=int, default=5, metavar="N",
                   help="beam search width (default: 5; 1 is faster, less accurate)")
    d.add_argument("--no-vad", action="store_true",
                   help="disable voice-activity filtering (VAD skips silence and cuts down "
                        "on hallucinated text)")
    d.add_argument("--word-timestamps", action="store_true",
                   help="also produce word-level timestamps (json output, cue splitting)")
    d.add_argument("--prompt", default=None, metavar="TEXT",
                   help="initial prompt that steers style (default: a short Persian sentence "
                        "with ZWNJ and punctuation; pass '' to disable)")
    d.add_argument("--no-context", action="store_true",
                   help="do not condition on previous text (use if output gets stuck repeating)")
    d.add_argument("--batch-size", type=int, default=0, metavar="N",
                   help="batched decoding for GPUs, e.g. 8; much faster on long files (default: off)")

    t = p.add_argument_group("Persian text clean-up")
    t.add_argument("--no-normalize", action="store_true",
                   help="keep raw Whisper output (no ی/ک fixes, ZWNJ or punctuation clean-up)")
    t.add_argument("--no-zwnj", action="store_true",
                   help="do not insert ZWNJ between affixes and stems (می‌، ‌ها، ‌ترین)")
    t.add_argument("--digits", choices=DIGIT_MODES, default="keep",
                   help="convert digits to persian (۱۲۳) or western (123) (default: keep)")

    v = p.add_argument_group("verbosity")
    v.add_argument("-q", "--quiet", action="store_true", help="no progress bar or live segments")
    v.add_argument("-v", "--verbose", action="store_true", help="debug output")
    return p


# ------------------------------------------------------------------ helpers


class _Formatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if record.levelno >= logging.WARNING:
            return f"{record.levelname.lower()}: {message}"
        return message


def setup_logging(verbose: bool, quiet: bool) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_Formatter("%(message)s"))
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=level, handlers=[handler])
    if not verbose:
        for name in ("faster_whisper", "huggingface_hub", "urllib3", "httpx", "httpcore"):
            logging.getLogger(name).setLevel(logging.WARNING)


def parse_formats(values: Optional[List[str]], stdout: bool) -> List[str]:
    if not values:
        return [] if stdout else list(DEFAULT_FORMATS)
    formats: List[str] = []
    for value in values:
        for fmt in value.split(","):
            fmt = fmt.strip().lower().lstrip(".")
            if not fmt:
                continue
            if fmt not in FORMATS:
                raise SystemExit(f"error: unknown format {fmt!r}; choose from {', '.join(FORMATS)}")
            if fmt not in formats:
                formats.append(fmt)
    return formats


def collect_inputs(raw_paths: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for raw in raw_paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            found = sorted(
                f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in MEDIA_EXTENSIONS
            )
            if not found:
                log.warning("no media files found in %s", path)
            files.extend(found)
        elif path.is_file():
            files.append(path)
        else:
            raise SystemExit(f"error: no such file or directory: {raw}")
    unique: List[Path] = []
    seen = set()
    for f in files:
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def print_models() -> None:
    print(f"{'model':<16}{'params':<9}{'GPU fp16 / CPU int8':<22}notes")
    for name, params, memory, note in MODELS:
        print(f"{name:<16}{params:<9}{memory:<22}{note}")
    print("\nAny CTranslate2 Whisper model directory or Hugging Face repo id also works.")


def _clock(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{secs:04.1f}"


# --------------------------------------------------------------------- model


def load_model_with_fallback(args: argparse.Namespace):
    device = resolve_device(args.device)
    compute_type = resolve_compute_type(device, args.compute_type)
    threads = args.threads or (max(1, (os.cpu_count() or 2) // 2) if device == "cpu" else 0)
    common = dict(cpu_threads=threads, download_root=args.model_dir, local_files_only=args.offline)

    log.info("loading model %s on %s (%s)...", args.model, device, compute_type)
    started = time.monotonic()
    try:
        model = load_model(args.model, device=device, compute_type=compute_type, **common)
        if device == "cuda":
            warm_up(model)
    except Exception as exc:  # noqa: BLE001 - we want to fall back on any GPU failure
        if device != "cuda" or args.device != "auto":
            raise
        log.warning("GPU initialisation failed (%s)", str(exc).splitlines()[0])
        log.warning("falling back to CPU; for GPU support run: pip install 'transcribe-fa[cuda]'")
        device = "cpu"
        compute_type = resolve_compute_type(device, args.compute_type)
        common["cpu_threads"] = args.threads or max(1, (os.cpu_count() or 2) // 2)
        model = load_model(args.model, device=device, compute_type=compute_type, **common)
    log.info("model ready in %.1fs", time.monotonic() - started)
    return model, device, compute_type


# ---------------------------------------------------------------------- main


def process_file(path: Path, model, args: argparse.Namespace, formats: List[str]) -> None:
    from tqdm import tqdm

    out_dir = args.output_dir or path.parent
    targets = {fmt: out_dir / f"{path.stem}.{fmt}" for fmt in formats}
    if args.skip_existing and targets and all(t.exists() for t in targets.values()):
        log.info("%s: outputs exist, skipping", path.name)
        return

    persian = args.task == "transcribe" and args.language == DEFAULT_LANGUAGE
    prompt = args.prompt if args.prompt is not None else (DEFAULT_PROMPT if persian else None)
    do_normalize = not args.no_normalize and args.task == "transcribe"
    norm_opts = dict(digits=args.digits, zwnj=not args.no_zwnj)

    audio = load_audio(path)
    duration = audio_duration(audio)
    log.info("%s: %s of audio", path.name, _clock(duration))

    bar = tqdm(
        total=round(duration, 1),
        unit="s",
        disable=args.quiet or not sys.stderr.isatty(),
        file=sys.stderr,
        leave=False,
        bar_format="{percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f}s [{elapsed}<{remaining}]",
    )

    def on_segment(segment: Segment) -> None:
        bar.n = min(duration, max(bar.n, segment.end))
        bar.refresh()
        text = segment.text
        if do_normalize and (persian or args.language == "auto"):
            text = normalize(text, **norm_opts)
        if args.stdout:
            tqdm.write(text, file=sys.stdout)
        elif not args.quiet:
            tqdm.write(f"[{_clock(segment.start)} --> {_clock(segment.end)}] {text}", file=sys.stderr)

    started = time.monotonic()
    try:
        transcript = transcribe(
            model,
            audio,
            source=str(path),
            model_name=args.model,
            language=None if args.language == "auto" else args.language,
            task=args.task,
            beam_size=args.beam_size,
            vad=not args.no_vad,
            word_timestamps=args.word_timestamps or args.max_cue_chars > 0,
            initial_prompt=prompt,
            condition_on_previous_text=not args.no_context,
            batch_size=args.batch_size,
            on_segment=on_segment,
        )
    finally:
        bar.close()
    elapsed = time.monotonic() - started

    if args.language == "auto":
        log.info("detected language: %s (%.0f%%)", transcript.language,
                 transcript.language_probability * 100)
    if do_normalize and transcript.language == DEFAULT_LANGUAGE:
        normalize_transcript(transcript, **norm_opts)

    subtitles = transcript
    if args.max_cue_chars > 0:
        cues = split_for_subtitles(transcript.segments, max_chars=args.max_cue_chars)
        subtitles = replace(transcript, segments=cues)
        if do_normalize and transcript.language == DEFAULT_LANGUAGE:
            normalize_transcript(subtitles, **norm_opts)

    if targets:
        out_dir.mkdir(parents=True, exist_ok=True)
    for fmt, target in targets.items():
        source = subtitles if fmt in ("srt", "vtt") else transcript
        write_transcript(source, fmt, target, rtl_mark=args.rtl_mark)

    speed = duration / elapsed if elapsed > 0 else 0.0
    if not targets:
        written = "stdout"
    elif len(targets) == 1:
        written = str(next(iter(targets.values())))
    else:
        written = f"{out_dir / path.stem}.{{{','.join(targets)}}}"
    log.info("%s: %d segments in %.1fs (%.1fx realtime) -> %s",
             path.name, len(transcript.segments), elapsed, speed, written)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        return _main(argv)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


def _main(argv: Optional[Sequence[str]]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose, args.quiet)

    if args.list_models:
        print_models()
        return 0
    if args.download_only:
        path = download_model(args.model, args.model_dir)
        log.info("model %s is available at %s", args.model, path)
        return 0

    formats = parse_formats(args.formats, args.stdout)
    inputs = collect_inputs(args.inputs)
    if not inputs:
        parser.error("no input files given (or none with a known media extension)")
    if args.max_cue_chars < 0 or args.beam_size < 1 or args.batch_size < 0:
        parser.error("--max-cue-chars, --beam-size and --batch-size must be non-negative")

    try:
        model, _device, _compute_type = load_model_with_fallback(args)
    except Exception as exc:  # noqa: BLE001
        log.error("could not load model %s: %s", args.model, exc)
        return 2

    failures = 0
    for index, path in enumerate(inputs, 1):
        if len(inputs) > 1:
            log.info("[%d/%d] %s", index, len(inputs), path)
        try:
            process_file(path, model, args, formats)
        except Exception as exc:  # noqa: BLE001 - keep going with the other files
            failures += 1
            log.error("%s: %s", path, exc)
            if args.verbose:
                log.exception("traceback")
    if failures:
        log.error("%d of %d files failed", failures, len(inputs))
    return 1 if failures else 0
