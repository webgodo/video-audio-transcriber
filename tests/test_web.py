"""Tests for the browser front end.

The app module imports gradio, which CI deliberately does not install (the
test matrix installs with --no-deps so it needs no wheels and no model), so
those tests skip unless the web extra is present. The entry point itself must
work without gradio, and that part is always tested.
"""

import pytest

from video_audio_transcriber import web


def test_entry_point_explains_how_to_install_the_extra(monkeypatch, capsys):
    monkeypatch.setattr(web, "__name__", web.__name__)  # keep the module importable

    def missing(*args, **kwargs):
        raise ImportError("No module named 'gradio'")

    import builtins

    real_import = builtins.__import__

    def no_gradio(name, *args, **kwargs):
        if name.endswith("app") or name == "gradio":
            missing()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_gradio)
    assert web.main() == 1
    err = capsys.readouterr().err
    assert "pip install" in err
    assert "video-audio-transcriber[web]" in err


gradio = pytest.importorskip("gradio", reason="web extra not installed")

from video_audio_transcriber.web.app import (  # noqa: E402  - after the skip guard
    RTL_LANGUAGES,
    Settings,
    build,
    command_line,
)


def test_the_page_shows_the_matching_command_line():
    assert command_line("large-v3", "keep", 0, True) == "vatfa recording.mp3 -f srt"
    assert "-m medium" in command_line("medium", "keep", 0, True)
    assert "--digits persian" in command_line("large-v3", "persian", 0, True)
    assert "--max-cue-chars 42" in command_line("large-v3", "keep", 42, True)
    assert "--no-normalize" in command_line("large-v3", "keep", 0, False)


def test_persian_is_among_the_right_to_left_languages():
    assert "fa" in RTL_LANGUAGES
    assert "en" not in RTL_LANGUAGES


def test_settings_default_to_the_unrestricted_local_experience(monkeypatch):
    monkeypatch.delenv("VATFA_WEB_MODEL", raising=False)
    monkeypatch.delenv("VATFA_WEB_MAX_SECONDS", raising=False)
    settings = Settings()
    assert settings.max_seconds == 0  # no clip-length cap locally
    assert len(settings.models) > 1


def test_settings_can_be_narrowed_by_environment(monkeypatch):
    monkeypatch.setenv("VATFA_WEB_MODEL", "small")
    monkeypatch.setenv("VATFA_WEB_MAX_SECONDS", "30")
    monkeypatch.setenv("VATFA_WEB_MODELS", "small")
    settings = Settings()
    assert settings.model == "small"
    assert settings.max_seconds == 30
    assert settings.models == ["small"]


def test_a_nonsense_environment_value_falls_back(monkeypatch):
    monkeypatch.setenv("VATFA_WEB_MAX_SECONDS", "not-a-number")
    assert Settings().max_seconds == 0


def test_the_page_builds():
    demo = build()
    assert demo.title.startswith("video-audio-transcriber")
