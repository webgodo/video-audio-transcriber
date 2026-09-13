"""A small Gradio front end over the same code the CLI uses.

Every control maps one to one onto a command-line flag, and the page shows
the equivalent command for whatever is selected. That is deliberate: the
browser is the easy way in, and the command line is where the work actually
gets done.

Text streams in as it decodes rather than appearing all at once, which is the
part worth seeing. That comes from :class:`~video_audio_transcriber.pipeline.SegmentStream`
wrapping the existing ``on_segment`` callback.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, List, Optional

import gradio as gr

from .. import __version__
from ..normalize import DIGIT_MODES, normalize
from ..pipeline import SegmentStream, finish
from ..transcriber import (
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    MODELS,
    SAMPLE_RATE,
    audio_duration,
    load_audio,
    load_model,
    resolve_compute_type,
    resolve_device,
    warm_up,
)
from ..writers import FORMATS, write_transcript

log = logging.getLogger("video_audio_transcriber")

DOWNLOAD_FORMATS = ("srt", "vtt", "txt", "json", "html")


def _env(name: str, default: str) -> str:
    return os.environ.get(f"VATFA_WEB_{name}", default)


def _int_env(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


class Settings:
    """Configuration, so one app serves both a local run and a locked-down one.

    Defaults are the unrestricted local experience. A deployment can narrow
    them through the environment without a second copy of this file.
    """

    def __init__(self) -> None:
        self.model = _env("MODEL", DEFAULT_MODEL)
        self.max_seconds = _int_env("MAX_SECONDS", 0)  # 0 = no limit
        self.beam_size = _int_env("BEAM_SIZE", 5)
        self.device = _env("DEVICE", "auto")
        self.compute_type = _env("COMPUTE_TYPE", "auto")
        allowed = _env("MODELS", "")
        self.models = [m.strip() for m in allowed.split(",") if m.strip()] or [n for n, *_ in MODELS]
        if self.model not in self.models:
            self.models.insert(0, self.model)


SETTINGS = Settings()
_MODEL: Any = None
_MODEL_NAME = ""
_LOCK = threading.Lock()


def get_model(name: str) -> Any:
    """Load the model once and reuse it, swapping only if the choice changes."""
    global _MODEL, _MODEL_NAME
    if _MODEL is not None and _MODEL_NAME == name:
        return _MODEL
    device = resolve_device(SETTINGS.device)
    compute_type = resolve_compute_type(device, SETTINGS.compute_type)
    log.info("loading model %s on %s (%s)...", name, device, compute_type)
    model = load_model(name, device=device, compute_type=compute_type)
    if device == "cuda":
        warm_up(model)
    _MODEL, _MODEL_NAME = model, name
    return model


def command_line(model: str, digits: str, cue_chars: int, clean: bool) -> str:
    parts = ["vatfa", "recording.mp3", "-f", "srt"]
    if model != DEFAULT_MODEL:
        parts += ["-m", model]
    if digits != "keep":
        parts += ["--digits", digits]
    if cue_chars:
        parts += ["--max-cue-chars", str(cue_chars)]
    if not clean:
        parts.append("--no-normalize")
    return " ".join(parts)


def _write_downloads(transcript: Any, subtitles: Any, media: Optional[str]) -> List[str]:
    out_dir = Path(tempfile.mkdtemp(prefix="vatfa-"))
    stem = Path(media).stem if media else "transcript"
    paths = []
    for fmt in DOWNLOAD_FORMATS:
        if fmt not in FORMATS:  # pragma: no cover - registry changed under us
            continue
        target = out_dir / f"{stem}.{fmt}"
        source = subtitles if fmt in ("srt", "vtt") else transcript
        write_transcript(source, fmt, target, media=media, embed_media=(fmt == "html"))
        paths.append(str(target))
    return paths


def transcribe_file(
    media: Optional[str],
    model_name: str,
    language: str,
    clean: bool,
    digits: str,
    cue_chars: int,
    use_prompt: bool,
):
    """Gradio event handler. A generator, so each yield repaints the page."""
    if not media:
        raise gr.Error("Choose an audio or video file first.")

    audio = load_audio(media)
    duration = audio_duration(audio)
    if SETTINGS.max_seconds and duration > SETTINGS.max_seconds:
        audio = audio[: SETTINGS.max_seconds * SAMPLE_RATE]
        gr.Warning(f"Trimmed to the first {SETTINGS.max_seconds}s for this demo.")
        duration = SETTINGS.max_seconds

    persian = language == DEFAULT_LANGUAGE
    options = dict(digits=digits, zwnj=True, punctuation=True)
    started = time.monotonic()

    with _LOCK:  # one decode at a time: the model is shared
        model = get_model(model_name)
        stream = SegmentStream(
            model,
            audio,
            source=media,
            model_name=model_name,
            language=None if language == "auto" else language,
            beam_size=SETTINGS.beam_size,
            vad=True,
            word_timestamps=True,
            initial_prompt=DEFAULT_PROMPT if (use_prompt and persian) else None,
        )
        lines: List[str] = []
        for segment in stream:
            text = normalize(segment.text, **options) if clean else segment.text
            lines.append(text)
            yield "\n".join(lines), gr.update(), gr.update()

        transcript = stream.transcript
        if transcript is None:  # pragma: no cover - cancelled before finishing
            return
        transcript, subtitles = finish(
            transcript, normalize=clean, max_cue_chars=cue_chars, **options
        )
        files = _write_downloads(transcript, subtitles, media)

    elapsed = time.monotonic() - started
    speed = f"{duration / elapsed:.1f}x realtime" if elapsed > 0 else ""
    summary = (
        f"**{len(transcript.segments)} segments** · {duration:.1f}s of audio in "
        f"{elapsed:.1f}s ({speed}) · language `{transcript.language}` · model `{model_name}`"
    )
    yield transcript.text, gr.update(value=files, visible=True), summary


#: Languages Whisper can produce that are written right to left.
RTL_LANGUAGES = frozenset({"fa", "ar", "he", "ur", "ps"})
LANGUAGES = ["fa", "auto", "en", "ar", "tr", "ps", "ur"]


def build() -> gr.Blocks:
    with gr.Blocks(title=f"video-audio-transcriber {__version__}", analytics_enabled=False) as demo:
        gr.Markdown(
            "# video-audio-transcriber\n"
            "Offline transcription with OpenAI Whisper, with Persian text done properly. "
            "Nothing is uploaded anywhere: this page talks to a model running on this machine."
        )
        with gr.Row():
            with gr.Column(scale=2):
                media = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio or video")
                with gr.Row():
                    model_name = gr.Dropdown(
                        SETTINGS.models, value=SETTINGS.model, label="Model",
                        interactive=len(SETTINGS.models) > 1,
                    )
                    language = gr.Dropdown(
                        LANGUAGES, value=DEFAULT_LANGUAGE,
                        label="Language", info="forcing fa avoids Arabic/Urdu misdetection",
                    )
                clean = gr.Checkbox(value=True, label="Persian clean-up (ی/ک, ZWNJ, ؟ ،)")
                use_prompt = gr.Checkbox(
                    value=True, label="Persian initial prompt",
                    info="nudges Whisper towards ZWNJ and Persian punctuation",
                )
                digits = gr.Radio(list(DIGIT_MODES), value="keep", label="Digits")
                cue_chars = gr.Slider(
                    0, 80, value=0, step=1, label="Split subtitle cues at N characters",
                    info="0 keeps Whisper's own segments",
                )
                run = gr.Button("Transcribe", variant="primary")
            with gr.Column(scale=3):
                # Gradio renders RTL natively, so no CSS override is needed.
                # Persian placeholder to match the default language: an
                # English sentence in an RTL box has its full stop reordered
                # to the left and reads as broken.
                text = gr.Textbox(
                    label="Transcript", lines=16, rtl=True, text_align="right",
                    placeholder="متن، همزمان با رمزگشایی، اینجا ظاهر می‌شود",
                )
                summary = gr.Markdown()
                files = gr.File(label="Download", file_count="multiple", visible=False)

        equivalent = gr.Markdown()

        def show_command(model: str, digits_mode: str, chars: int, cleaning: bool) -> str:
            command = command_line(model, digits_mode, int(chars), cleaning)
            return f"Same thing on the command line:\n```bash\n{command}\n```"

        inputs = [media, model_name, language, clean, digits, cue_chars, use_prompt]
        run.click(transcribe_file, inputs=inputs, outputs=[text, files, summary])
        command_inputs = [model_name, digits, cue_chars, clean]
        for control in command_inputs:
            control.change(show_command, inputs=command_inputs, outputs=equivalent)

        def flip_direction(chosen: str):
            rtl = chosen in RTL_LANGUAGES or chosen == "auto"
            return gr.update(
                rtl=rtl,
                text_align="right" if rtl else "left",
                placeholder=(
                    "متن، همزمان با رمزگشایی، اینجا ظاهر می‌شود"
                    if rtl
                    else "Text appears here as it is decoded."
                ),
            )

        language.change(flip_direction, inputs=language, outputs=text)
        demo.load(lambda: show_command(SETTINGS.model, "keep", 0, True), outputs=equivalent)
    return demo


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vatfa-web", description="Browser interface for video-audio-transcriber."
    )
    parser.add_argument("--host", default="127.0.0.1", help="default: 127.0.0.1 (this machine only)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a public Gradio share link")
    parser.add_argument("--open", action="store_true", help="open a browser window")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    demo = build()
    demo.queue(default_concurrency_limit=1, max_size=16)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.open,
        max_file_size="512mb",
    )
    return 0
