import pytest

from video_audio_transcriber.normalize import ZWNJ
from video_audio_transcriber.pipeline import SegmentStream, finish
from video_audio_transcriber.transcriber import Segment, Transcript, Word


class FakeSegment:
    """The shape faster-whisper yields, which transcribe() adapts."""

    def __init__(self, start, end, text, words=None):
        self.start, self.end, self.text = start, end, text
        self.words = words
        self.avg_logprob, self.no_speech_prob = -0.2, 0.01


class FakeInfo:
    language, language_probability, duration = "fa", 0.99, 3.0


class FakeModel:
    """A model whose transcribe() yields lazily, like the real one."""

    def __init__(self, texts, fail_at=None):
        self.texts, self.fail_at = texts, fail_at

    def transcribe(self, audio, **options):
        def generate():
            for i, text in enumerate(self.texts):
                if self.fail_at == i:
                    raise RuntimeError("decode blew up")
                yield FakeSegment(float(i), float(i) + 1, text)

        return generate(), FakeInfo()


def test_segments_arrive_in_order_and_the_transcript_is_set():
    stream = SegmentStream(FakeModel(["یک", "دو", "سه"]), None)
    assert [s.text for s in stream] == ["یک", "دو", "سه"]
    assert stream.transcript is not None
    assert len(stream.transcript.segments) == 3
    assert stream.transcript.language == "fa"


def test_a_decode_error_surfaces_instead_of_hanging():
    stream = SegmentStream(FakeModel(["یک", "دو", "سه"], fail_at=1), None)
    with pytest.raises(RuntimeError, match="decode blew up"):
        list(stream)


def test_cancel_stops_early_without_deadlocking():
    # The queue is bounded, so decoding can only run LOOKAHEAD segments past
    # the consumer. Without that bound the worker would finish the whole file
    # before a cancel could land, which is exactly what a browser tab closing
    # mid-transcription must not cause.
    stream = SegmentStream(FakeModel([str(i) for i in range(500)]), None)
    seen = []
    for segment in stream:
        seen.append(segment)
        if len(seen) == 3:
            stream.cancel()
    assert 3 <= len(seen) <= 3 + SegmentStream.LOOKAHEAD + 1


def test_a_consumer_that_walks_away_does_not_wedge_the_worker():
    stream = SegmentStream(FakeModel([str(i) for i in range(500)]), None)
    iterator = iter(stream)
    next(iterator)
    stream.cancel()
    del iterator
    stream._thread.join(timeout=5)
    assert not stream._thread.is_alive()


def test_a_stream_cannot_be_iterated_twice():
    stream = SegmentStream(FakeModel(["یک"]), None)
    list(stream)
    with pytest.raises(RuntimeError, match="only be iterated once"):
        list(stream)


def make_transcript(text="من می روم", words=None):
    return Transcript(
        source="a.mp3", language="fa", language_probability=1.0, duration=2.0, model="tiny",
        segments=[Segment(0, 0.0, 2.0, text, words=words or [])],
    )


def test_finish_normalizes_and_returns_one_object_without_cue_splitting():
    transcript, subtitles = finish(make_transcript())
    assert transcript.segments[0].text == f"من می{ZWNJ}روم"
    assert subtitles is transcript


def test_finish_can_skip_normalizing():
    transcript, _ = finish(make_transcript(), normalize=False)
    assert transcript.segments[0].text == "من می روم"


def test_finish_splits_cues_into_a_separate_copy():
    words = [Word(i * 0.5, i * 0.5 + 0.4, f" کلمه{i}", 0.9) for i in range(8)]
    transcript, subtitles = finish(
        make_transcript(" ".join(w.word.strip() for w in words), words), max_cue_chars=17
    )
    assert subtitles is not transcript
    assert len(subtitles.segments) > len(transcript.segments)
    assert len(transcript.segments) == 1  # the original is left whole


def test_finish_leaves_other_languages_alone():
    english = Transcript(source="a.mp3", language="en", language_probability=1.0, duration=1.0,
                         model="tiny", segments=[Segment(0, 0.0, 1.0, "he said , hello ?")])
    transcript, _ = finish(english)
    assert transcript.segments[0].text == "he said , hello ?"
