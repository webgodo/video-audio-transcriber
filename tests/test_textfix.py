from video_audio_transcriber.normalize import ZWNJ
from video_audio_transcriber.textfix import (
    detect_kind,
    fix_srt,
    fix_text,
    fix_txt,
    fix_vtt,
    read_text,
)

SRT = (
    "1\n"
    "00:00:01,000 --> 00:00:04,000\n"
    "من می روم\n"
    "\n"
    "2\n"
    "00:00:04,500 --> 00:00:06,000\n"
    "كتاب ها را خواندم?\n"
)

VTT = (
    "WEBVTT\n"
    "\n"
    "NOTE می رود is left alone here\n"
    "\n"
    "cue-1\n"
    "00:00:01.000 --> 00:00:02.000 align:start\n"
    "می رود\n"
)


def test_detect_kind_by_suffix():
    assert detect_kind("anything", "a.srt") == "srt"
    assert detect_kind("anything", "a.VTT") == "vtt"
    assert detect_kind("anything", "a.txt") == "txt"


def test_detect_kind_by_content():
    assert detect_kind(VTT) == "vtt"
    assert detect_kind(SRT) == "srt"
    assert detect_kind("سلام، حال شما چطور است؟") == "txt"


def test_srt_indices_and_timestamps_are_untouched():
    out = fix_srt(SRT)
    for line in ("1", "2", "00:00:01,000 --> 00:00:04,000", "00:00:04,500 --> 00:00:06,000"):
        assert line in out.split("\n")


def test_srt_cue_text_is_normalized():
    lines = fix_srt(SRT).split("\n")
    assert lines[2] == f"من می{ZWNJ}روم"
    assert lines[6] == f"کتاب{ZWNJ}ها را خواندم؟"


def test_srt_block_count_is_preserved():
    assert len(fix_srt(SRT).split("\n")) == len(SRT.split("\n"))


def test_vtt_header_identifier_and_notes_are_preserved():
    lines = fix_vtt(VTT).split("\n")
    assert lines[0] == "WEBVTT"
    assert lines[2] == "NOTE می رود is left alone here"  # metadata, not cue text
    assert lines[4] == "cue-1"
    assert lines[5] == "00:00:01.000 --> 00:00:02.000 align:start"
    assert lines[6] == f"می{ZWNJ}رود"


def test_crlf_line_endings_survive():
    out = fix_srt(SRT.replace("\n", "\r\n"))
    assert out.count("\r\n") == SRT.count("\n")
    assert "\n" not in out.replace("\r\n", "")  # no bare LF introduced


def test_lf_input_stays_lf():
    assert "\r" not in fix_srt(SRT)


def test_blocks_without_a_timing_line_are_left_alone():
    broken = "1\nمی رود\n"
    assert fix_srt(broken) == broken


def test_txt_preserves_blank_lines():
    text = "می رود\n\nكتاب ها\n"
    assert fix_txt(text) == f"می{ZWNJ}رود\n\nکتاب{ZWNJ}ها\n"


def test_latin_text_is_untouched():
    text = "Hello, world? Yes; 1,000.\n"
    assert fix_txt(text) == text


def test_options_are_forwarded_to_normalize():
    assert fix_txt("می رود\n", zwnj=False) == "می رود\n"
    assert fix_txt("سال 1403\n", digits="persian") == "سال ۱۴۰۳\n"
    assert fix_txt("خوبی?\n", punctuation=False) == "خوبی?\n"


def test_fix_text_detects_and_dispatches():
    assert fix_text(SRT, name="a.srt") == fix_srt(SRT)
    assert fix_text(VTT) == fix_vtt(VTT)


def test_fix_text_rejects_an_unknown_kind():
    try:
        fix_text("x", kind="docx")
    except ValueError as exc:
        assert "docx" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_read_text_strips_a_utf8_bom(tmp_path):
    path = tmp_path / "a.srt"
    path.write_bytes("﻿می رود".encode("utf-8"))
    assert read_text(path) == "می رود"


def test_read_text_falls_back_to_cp1256(tmp_path):
    # cp1256 has no Persian yeh or keheh at all, so a legacy Persian subtitle
    # file is necessarily spelled with the Arabic letters. Decoding it is half
    # the job; the normalizer does the other half.
    path = tmp_path / "b.srt"
    path.write_bytes("مي رود".encode("cp1256"))
    assert read_text(path) == "مي رود"
    assert fix_txt(read_text(path)) == f"می{ZWNJ}رود"


def test_empty_input():
    assert fix_txt("") == ""
    assert fix_srt("") == ""
