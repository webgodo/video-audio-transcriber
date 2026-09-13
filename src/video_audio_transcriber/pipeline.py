"""Turn the blocking transcribe() call into something you can iterate.

``transcribe()`` runs to completion and reports progress by calling
``on_segment`` from inside its own loop. That suits a CLI, which just prints
each line as it arrives. A web handler needs the opposite shape: something it
can pull from and yield.

:class:`SegmentStream` bridges the two with a worker thread and a queue. It is
a class rather than a generator function because the caller needs both the
live segments *and* the finished :class:`Transcript`, and smuggling a return
value out of a generator through ``StopIteration`` reads badly at the call
site.

Nothing here imports the model, a web framework, or anything else new, so it
can be tested against a fake model in milliseconds.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import replace
from typing import Any, Iterator, Optional, Tuple

from .normalize import normalize_transcript
from .transcriber import DEFAULT_LANGUAGE, Segment, Transcript, transcribe
from .writers import split_for_subtitles

log = logging.getLogger("video_audio_transcriber")


class _Cancelled(Exception):
    """Raised inside the segment callback to unwind the worker thread."""


class _Done:
    """Sentinel put on the queue when decoding finishes."""

    __slots__ = ("error",)

    def __init__(self, error: Optional[BaseException] = None) -> None:
        self.error = error


class SegmentStream:
    """Iterate segments as they decode; ``transcript`` is set when it ends.

    >>> stream = SegmentStream(model, audio, language="fa")
    >>> for segment in stream:
    ...     show(segment.text)
    >>> save(stream.transcript)

    Cancellation is free: ``transcribe()`` calls the callback inside its own
    loop with no exception handling, so raising from the callback aborts
    decoding immediately. Without that, a viewer who closes their browser tab
    would leave a thread pinning a CPU core until the whole file finished.
    """

    #: How far decoding may run ahead of whoever is reading. Bounded so a slow
    #: consumer applies backpressure and a cancel takes effect within a segment
    #: or two, rather than after the whole file has already been decoded.
    LOOKAHEAD = 8

    def __init__(self, model: Any, audio: Any, **options: Any) -> None:
        self.transcript: Optional[Transcript] = None
        self._queue: "queue.Queue" = queue.Queue(maxsize=self.LOOKAHEAD)
        self._cancel = threading.Event()
        self._model = model
        self._audio = audio
        self._options = options
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False

    def _on_segment(self, segment: Segment) -> None:
        # Poll rather than block forever on put(): a consumer that walks away
        # without cancelling would otherwise wedge this thread on a full queue.
        while True:
            if self._cancel.is_set():
                raise _Cancelled
            try:
                self._queue.put(segment, timeout=0.1)
                return
            except queue.Full:
                continue

    def _put_done(self, done: "_Done") -> None:
        """Deliver the end-of-stream sentinel without wedging this thread.

        The queue is bounded, so a plain put() would block forever when the
        consumer has stopped reading, which is exactly the situation after a
        cancel. If we have been cancelled, drop a queued segment to make room:
        nobody is going to read it.
        """
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                self._queue.put(done, timeout=0.1)
                return
            except queue.Full:
                if self._cancel.is_set():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:  # pragma: no cover - lost a race, retry
                        pass
        log.debug("segment stream consumer stopped reading; worker giving up")

    def _run(self) -> None:
        try:
            self.transcript = transcribe(
                self._model, self._audio, on_segment=self._on_segment, **self._options
            )
        except _Cancelled:
            self._put_done(_Done())
            return
        except BaseException as exc:  # noqa: BLE001 - re-raised in the consumer
            self._put_done(_Done(exc))
            return
        self._put_done(_Done())

    def __iter__(self) -> Iterator[Segment]:
        if self._started:
            raise RuntimeError("a SegmentStream can only be iterated once")
        self._started = True
        self._thread.start()
        while True:
            item = self._queue.get()
            if isinstance(item, _Done):
                if item.error is not None:
                    raise item.error
                return
            yield item

    def cancel(self) -> None:
        """Ask decoding to stop at the next segment boundary."""
        self._cancel.set()


def finish(
    transcript: Transcript,
    *,
    normalize: bool = True,
    digits: str = "keep",
    zwnj: bool = True,
    punctuation: bool = True,
    max_cue_chars: int = 0,
) -> Tuple[Transcript, Transcript]:
    """Apply the Persian clean-up and subtitle splitting to a raw transcript.

    Returns ``(transcript, subtitles)``. They are the same object unless cue
    splitting produced a second, re-chunked copy for ``srt``/``vtt``.
    """
    options = dict(digits=digits, zwnj=zwnj, punctuation=punctuation)
    persian = transcript.language == DEFAULT_LANGUAGE
    if normalize and persian:
        normalize_transcript(transcript, **options)

    if max_cue_chars <= 0:
        return transcript, transcript

    cues = split_for_subtitles(transcript.segments, max_chars=max_cue_chars)
    subtitles = replace(transcript, segments=cues)
    if normalize and persian:
        normalize_transcript(subtitles, **options)
    return transcript, subtitles
