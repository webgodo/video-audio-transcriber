import io
import json

from video_audio_transcriber.transcriber import Segment, Transcript, Word
from video_audio_transcriber.writers import (
    RLM,
    format_timestamp,
    split_for_subtitles,
    write_json,
    write_srt,
    write_tsv,
    write_txt,
    write_vtt,
)


def make_transcript():
    return Transcript(
        source="a.mp3", language="fa", language_probability=0.99, duration=5.0, model="large-v3",
        segments=[Segment(0, 0.0, 1.5, "سلام"), Segment(1, 1.5, 3.2567, "خوبی؟")],
    )


def render(writer, **options):
    fh = io.StringIO()
    writer(make_transcript(), fh, **options)
    return fh.getvalue()


def test_format_timestamp():
    assert format_timestamp(3661.5) == "01:01:01,500"
    assert format_timestamp(0.0004) == "00:00:00,000"
    assert format_timestamp(1.5, ".") == "00:00:01.500"
    assert format_timestamp(-1) == "00:00:00,000"


def test_srt():
    assert render(write_srt) == (
        "1\n00:00:00,000 --> 00:00:01,500\nسلام\n\n"
        "2\n00:00:01,500 --> 00:00:03,257\nخوبی؟\n\n"
    )


def test_srt_rtl_mark():
    assert f"\n{RLM}سلام\n" in render(write_srt, rtl_mark=True)


def test_vtt():
    out = render(write_vtt)
    assert out.startswith("WEBVTT\n\n")
    assert "00:00:00.000 --> 00:00:01.500\nسلام\n" in out


def test_txt():
    assert render(write_txt) == "سلام\nخوبی؟\n"


def test_tsv():
    assert render(write_tsv) == "start\tend\ttext\n0\t1500\tسلام\n1500\t3257\tخوبی؟\n"


def test_json():
    data = json.loads(render(write_json))
    assert data["text"] == "سلام\nخوبی؟"
    assert data["language"] == "fa"
    assert data["segments"][1]["end"] == 3.257
    assert "words" not in data["segments"][0]


def test_json_includes_words_when_present():
    transcript = make_transcript()
    transcript.segments[0].words = [Word(0.0, 0.5, " سلام", 0.9)]
    fh = io.StringIO()
    write_json(transcript, fh)
    assert json.loads(fh.getvalue())["segments"][0]["words"][0]["word"] == "سلام"


def test_split_for_subtitles_by_length():
    words = [Word(i * 0.5, i * 0.5 + 0.4, f" کلمه{i}", 0.9) for i in range(10)]
    segment = Segment(0, 0.0, 5.0, " ".join(w.word.strip() for w in words), words=words)
    cues = split_for_subtitles([segment], max_chars=17)
    assert [c.text for c in cues] == [
        "کلمه0 کلمه1 کلمه2", "کلمه3 کلمه4 کلمه5", "کلمه6 کلمه7 کلمه8", "کلمه9",
    ]
    assert (cues[0].start, cues[0].end) == (0.0, 1.4)
    assert [c.id for c in cues] == [0, 1, 2, 3]


def test_split_for_subtitles_on_pause():
    words = [Word(0.0, 0.4, " الف", 0.9), Word(2.0, 2.4, " ب", 0.9)]
    segment = Segment(0, 0.0, 2.4, "الف ب", words=words)
    cues = split_for_subtitles([segment], max_chars=100, max_duration=1.0)
    assert [c.text for c in cues] == ["الف", "ب"]


def test_split_for_subtitles_passthrough():
    long_no_words = Segment(0, 0.0, 30.0, "x" * 100)
    short = Segment(1, 30.0, 31.0, "کوتاه", words=[Word(30.0, 31.0, " کوتاه", 0.9)])
    assert split_for_subtitles([long_no_words, short], max_chars=10) == [long_no_words, short]


def test_format_registry_stays_in_sync():
    from video_audio_transcriber.writers import CUE_FORMATS, FORMATS, WRITERS

    assert set(FORMATS) == set(WRITERS)
    assert tuple(FORMATS) == tuple(WRITERS)  # order is the order offered on the CLI
    assert CUE_FORMATS <= set(FORMATS)
