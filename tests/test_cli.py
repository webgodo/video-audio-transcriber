import io

from video_audio_transcriber.cli import force_utf8, program_name


def test_force_utf8_rewrites_a_legacy_stream():
    # Regression test for the Windows default console encoding: cp1252 cannot
    # encode any Persian letter, so --help and --stdout both died on it.
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    force_utf8(stream)
    assert stream.encoding.lower().replace("-", "") == "utf8"
    stream.write("می‌رود")
    stream.flush()
    assert "می‌رود".encode("utf-8") in raw.getvalue()


def test_force_utf8_ignores_streams_that_cannot_be_reconfigured():
    force_utf8(io.StringIO(), None, object())  # must not raise


def test_program_name_falls_back_for_module_and_repl(monkeypatch):
    monkeypatch.setattr("sys.argv", ["/usr/bin/vatfa"])
    assert program_name() == "vatfa"
    monkeypatch.setattr("sys.argv", ["/usr/bin/video-audio-transcriber"])
    assert program_name() == "video-audio-transcriber"
    monkeypatch.setattr("sys.argv", ["__main__.py"])
    assert program_name() == "vatfa"
    monkeypatch.setattr("sys.argv", [""])
    assert program_name() == "vatfa"
