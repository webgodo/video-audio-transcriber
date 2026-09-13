"""Thin wrapper around faster-whisper (CTranslate2 port of OpenAI Whisper).

The Whisper weights are OpenAI's open-source checkpoints, converted to the
CTranslate2 format. Everything runs locally; the only network access is the
one-time model download from the Hugging Face Hub.
"""

from __future__ import annotations

import ctypes
import glob
import importlib
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

log = logging.getLogger("video_audio_transcriber")

SAMPLE_RATE = 16_000
DEFAULT_MODEL = "large-v3"
DEFAULT_LANGUAGE = "fa"
# Whisper treats the initial prompt as "previous text", so a well-formed
# Persian sentence nudges it towards ZWNJs and Persian punctuation.
DEFAULT_PROMPT = "متن زیر به فارسی است. نیم‌فاصله‌ها و علائم نگارشی، مانند ویرگول و نقطه، رعایت شده‌اند."

# (name, parameters, approx. memory GPU fp16 / CPU int8, note)
MODELS = (
    ("tiny", "39 M", "0.5 GB / 0.3 GB", "very fast; poor Persian accuracy"),
    ("base", "74 M", "0.6 GB / 0.4 GB", "poor Persian accuracy"),
    ("small", "244 M", "1.0 GB / 0.6 GB", "usable for clear studio speech"),
    ("medium", "769 M", "2.5 GB / 1.5 GB", "good; a sensible choice on CPU-only machines"),
    ("large-v2", "1550 M", "4.5 GB / 3.0 GB", "excellent"),
    ("large-v3", "1550 M", "4.5 GB / 3.0 GB", "best Persian accuracy (default)"),
    ("large-v3-turbo", "809 M", "2.5 GB / 1.6 GB", "~4x faster than large-v3, slightly less accurate"),
)


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0


@dataclass
class Transcript:
    source: str
    language: str
    language_probability: float
    duration: float
    model: str
    segments: List[Segment] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(s.text for s in self.segments if s.text.strip())


# ------------------------------------------------------------------ CUDA libs

_nvidia_libs_loaded = False


def _preload_nvidia_libs() -> None:
    """Make pip-installed CUDA libraries visible to CTranslate2.

    ``pip install nvidia-cublas-cu12`` (the ``[cuda]`` extra) puts the shared
    libraries inside site-packages, where the dynamic loader does not look.
    Loading them here with RTLD_GLOBAL lets CTranslate2's later dlopen() by
    soname succeed without touching LD_LIBRARY_PATH. Harmless on CPU-only
    machines and on systems that already have CUDA installed. cuDNN is
    handled too, for CTranslate2 builds that still use it.
    """
    global _nvidia_libs_loaded
    if _nvidia_libs_loaded or sys.platform != "linux":
        return
    _nvidia_libs_loaded = True
    lib_dirs = []
    for package in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
        try:
            module = importlib.import_module(package)
        except Exception:
            continue
        # These are namespace packages (no __init__.py), so use __path__.
        lib_dirs.extend(str(d) for d in getattr(module, "__path__", []))
    pending = [p for d in lib_dirs for p in sorted(glob.glob(os.path.join(d, "*.so*")))]
    # Libraries depend on each other (libcublas needs libcublasLt, the cuDNN
    # sub-libraries need each other); a few passes resolve the ordering.
    for _ in range(4):
        failed = []
        for path in pending:
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(path)
        if not failed or len(failed) == len(pending):
            break
        pending = failed
    if lib_dirs:
        log.debug("preloaded NVIDIA libraries from %s", ", ".join(lib_dirs))


def cuda_available() -> bool:
    _preload_nvidia_libs()
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # pragma: no cover - depends on the machine
        return False


def resolve_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda" if cuda_available() else "cpu"
    if device == "cuda":
        _preload_nvidia_libs()
    return device


def resolve_compute_type(device: str, compute_type: str = "auto") -> str:
    if compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


# ---------------------------------------------------------------------- model


def load_model(
    name: str = DEFAULT_MODEL,
    *,
    device: str = "cpu",
    compute_type: str = "int8",
    cpu_threads: int = 0,
    download_root: Optional[Union[str, Path]] = None,
    local_files_only: bool = False,
) -> Any:
    """Load (downloading on first use) a faster-whisper ``WhisperModel``."""
    if device == "cuda":
        _preload_nvidia_libs()
    from faster_whisper import WhisperModel

    return WhisperModel(
        name,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        download_root=str(download_root) if download_root else None,
        local_files_only=local_files_only,
    )


#: Languages Whisper regularly returns for Persian speech. Sharing a script
#: with Persian, they are what a misdetection looks like, so seeing one of
#: these is worth telling the user about.
PERSIAN_LOOKALIKES = frozenset({"ar", "ur", "ps", "tg", "ckb", "sd"})


def detect_language(model: Any, audio: Any, *, vad: bool = True) -> Tuple[str, float]:
    """Identify the spoken language before decoding.

    Decoding needs to know the language up front, because the initial prompt
    is chosen from it and the prompt measurably changes Persian output. One
    extra forward pass over the opening seconds is cheap next to that.
    """
    language, probability, _ = model.detect_language(audio=audio, vad_filter=vad)
    return language, float(probability)


def warm_up(model: Any) -> None:
    """Run one tiny decode so GPU/library problems surface before real work."""
    import numpy as np

    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    segments, _ = model.transcribe(
        silence, language="fa", beam_size=1, vad_filter=False, without_timestamps=True
    )
    for _ in segments:
        pass


def download_model(name: str, download_root: Optional[Union[str, Path]] = None) -> str:
    from faster_whisper import download_model as _download

    return _download(name, cache_dir=str(download_root) if download_root else None)


# ---------------------------------------------------------------------- audio


def load_audio(source: Union[str, Path]) -> Any:
    """Decode any audio or video file to 16 kHz mono float32 (via PyAV/FFmpeg)."""
    from faster_whisper.audio import decode_audio

    return decode_audio(str(source), sampling_rate=SAMPLE_RATE)


def audio_duration(audio: Any) -> float:
    return len(audio) / SAMPLE_RATE


# ----------------------------------------------------------------- transcribe

SegmentCallback = Callable[[Segment], None]


def transcribe(
    model: Any,
    audio: Any,
    *,
    source: str = "",
    model_name: str = "",
    language: Optional[str] = DEFAULT_LANGUAGE,
    task: str = "transcribe",
    beam_size: int = 5,
    vad: bool = True,
    word_timestamps: bool = False,
    initial_prompt: Optional[str] = None,
    condition_on_previous_text: bool = True,
    batch_size: int = 0,
    on_segment: Optional[SegmentCallback] = None,
) -> Transcript:
    """Transcribe decoded audio (or a path) and return a :class:`Transcript`.

    ``language=None`` lets Whisper detect the language. ``batch_size > 0``
    switches to faster-whisper's batched pipeline (faster on GPUs).
    ``on_segment`` is called as each segment is decoded, for live output.
    """
    options = dict(
        language=language,
        task=task,
        beam_size=beam_size,
        vad_filter=vad,
        word_timestamps=word_timestamps,
        initial_prompt=initial_prompt or None,
        condition_on_previous_text=condition_on_previous_text,
    )
    if batch_size > 0:
        from faster_whisper import BatchedInferencePipeline

        runner = BatchedInferencePipeline(model)
        options["batch_size"] = batch_size
    else:
        runner = model

    segment_iter, info = runner.transcribe(audio, **options)
    transcript = Transcript(
        source=str(source),
        language=info.language,
        language_probability=float(info.language_probability),
        duration=float(info.duration),
        model=model_name,
    )
    for index, raw in enumerate(segment_iter):
        words = [
            Word(float(w.start), float(w.end), w.word, float(w.probability))
            for w in (raw.words or [])
        ]
        segment = Segment(
            id=index,
            start=float(raw.start),
            end=float(raw.end),
            text=raw.text.strip(),
            words=words,
            avg_logprob=float(raw.avg_logprob),
            no_speech_prob=float(raw.no_speech_prob),
        )
        transcript.segments.append(segment)
        if on_segment is not None:
            on_segment(segment)
    return transcript
