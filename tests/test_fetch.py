from pathlib import Path

import pytest

from video_audio_transcriber import fetch as fetch_mod
from video_audio_transcriber.fetch import is_url, resolve_urls


@pytest.mark.parametrize(
    "raw",
    ["https://soundcloud.com/a/b", "http://example.com/x.mp3", "HTTPS://EXAMPLE.COM/a",
     "file:///tmp/a.mp3", "  https://example.com/a  "],
)
def test_urls_are_recognised(raw):
    assert is_url(raw)


@pytest.mark.parametrize(
    "raw",
    ["./a.mp4", "a.mp4", "relative/path.mp3", "/abs/path.mp3", "~/a.mp3",
     r"C:\media\a.mp4", "C:/media/a.mp4", ""],
)
def test_paths_are_not_mistaken_for_urls(raw):
    # A Windows drive letter is the trap here: "C:" looks like a scheme until
    # you require the slashes.
    assert not is_url(raw)


def test_resolve_urls_passes_plain_paths_straight_through(tmp_path):
    inputs = ["a.mp3", "dir/b.wav", "/abs/c.m4a"]
    assert resolve_urls(inputs, tmp_path) == inputs


def test_resolve_urls_replaces_a_url_with_the_downloaded_path(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_mod, "fetch", lambda url, dest, **kw: Path(dest) / "got.m4a")
    out = resolve_urls(["first.mp3", "https://example.com/x"], tmp_path)
    assert out == ["first.mp3", str(tmp_path / "got.m4a")]


def test_one_bad_url_does_not_stop_the_batch(tmp_path, monkeypatch, caplog):
    def flaky(url, dest, **kw):
        if "bad" in url:
            raise RuntimeError("Video unavailable\nsecond line should not be logged")
        return Path(dest) / "ok.m4a"

    monkeypatch.setattr(fetch_mod, "fetch", flaky)
    seen = []
    out = resolve_urls(
        ["https://example.com/bad", "https://example.com/good", "local.mp3"],
        tmp_path,
        on_error=lambda url, exc: seen.append(url),
    )
    assert out == [str(tmp_path / "ok.m4a"), "local.mp3"]
    assert seen == ["https://example.com/bad"]
    assert "Video unavailable" in caplog.text
    assert "second line" not in caplog.text  # only the first line is reported


def test_missing_yt_dlp_gives_an_actionable_message(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_yt_dlp(name, *args, **kwargs):
        if name == "yt_dlp":
            raise ImportError("no module named yt_dlp")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_yt_dlp)
    with pytest.raises(RuntimeError) as caught:
        fetch_mod._yt_dlp()
    assert "pip install" in str(caught.value)
