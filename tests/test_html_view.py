import io
import json
import re

from video_audio_transcriber.html_view import render_html
from video_audio_transcriber.transcriber import Segment, Transcript, Word
from video_audio_transcriber.writers import write_html

PAYLOAD = re.compile(r'<script type="application/json" id="data">(.*?)</script>', re.S)


def make_transcript(language="fa", with_words=True, text="سلام دنیا"):
    words = [Word(0.0, 0.5, " سلام", 0.9), Word(0.5, 1.2, " دنیا", 0.8)] if with_words else []
    return Transcript(
        source="/home/someone/private/clients/acme/call.mp3",
        language=language,
        language_probability=0.99,
        duration=1.2,
        model="large-v3",
        segments=[Segment(0, 0.0, 1.2, text, words=words)],
    )


def payload(page):
    match = PAYLOAD.search(page)
    assert match, "no json payload in the page"
    return json.loads(match.group(1).replace("<\\/", "</"))


def test_page_makes_no_external_requests():
    # The whole claim of this project is that nothing leaves your machine. A
    # font CDN or a script tag pointing outward would quietly make that false.
    page = render_html(make_transcript(), media="call.mp3")
    assert "http://" not in page
    assert "https://" not in page


def test_payload_round_trips():
    data = payload(render_html(make_transcript()))
    assert data["segments"][0]["text"] == "سلام دنیا"
    assert [w["word"] for w in data["segments"][0]["words"]] == ["سلام", "دنیا"]


def test_payload_survives_a_closing_script_tag_in_the_transcript():
    page = render_html(make_transcript(text="</script><b>x</b> سلام"))
    assert "</script><b>" not in page.split('id="data">')[1].split("</script>")[0]
    assert payload(page)["segments"][0]["text"] == "</script><b>x</b> سلام"


def test_persian_is_right_to_left():
    page = render_html(make_transcript("fa"))
    assert '<html lang="fa" dir="rtl">' in page


def test_english_is_left_to_right():
    page = render_html(make_transcript("en", text="hello world"))
    assert '<html lang="en" dir="ltr">' in page
    assert "Search…" in page


def test_absolute_source_path_is_not_leaked():
    page = render_html(make_transcript())
    assert "/home/someone/private" not in page
    assert payload(page)["source"] == "call.mp3"


def test_media_is_linked_relative_to_the_output_file():
    page = render_html(make_transcript(), media="/tmp/media/call.mp3", dest="/tmp/out/call.html")
    assert 'src="../media/call.mp3"' in page
    assert "<audio" in page


def test_video_media_gets_a_video_tag():
    page = render_html(make_transcript(), media="clip.mp4")
    assert "<video" in page


def test_no_media_renders_without_a_player():
    page = render_html(make_transcript())
    assert "<audio" not in page and "<video" not in page


def test_embedded_media_becomes_a_data_uri(tmp_path):
    clip = tmp_path / "a.mp3"
    clip.write_bytes(b"ID3fake-audio-bytes")
    page = render_html(make_transcript(), media=clip, embed_media=True)
    assert "src=\"data:audio/mpeg;base64," in page
    assert str(tmp_path) not in page


def test_segments_without_word_timestamps_still_render():
    page = render_html(make_transcript(with_words=False))
    assert payload(page)["segments"][0]["text"] == "سلام دنیا"
    assert "words" not in payload(page)["segments"][0]


def test_empty_transcript_renders():
    empty = Transcript(source="a.mp3", language="fa", language_probability=1.0,
                       duration=0.0, model="tiny", segments=[])
    page = render_html(empty)
    assert payload(page)["segments"] == []


def test_write_html_goes_through_the_writer_registry():
    fh = io.StringIO()
    write_html(make_transcript(), fh, media="call.mp3")
    assert fh.getvalue().startswith("<!doctype html>")
