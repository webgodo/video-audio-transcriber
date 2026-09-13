# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
uses [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-09-14

The release that turned a working script into a project.

### Added

- `--text` runs the Persian clean-up over `.txt`, `.srt` and `.vtt` files with
  no model and no download, so it also works on subtitles you already have.
  Timestamps, cue indices, WebVTT headers and `NOTE` blocks are preserved byte
  for byte, and legacy windows-1256 files are decoded automatically.
- `-f html` writes a single self-contained interactive page: click a line and
  the audio seeks there, words highlight as it plays, and there is a search
  box. It makes no external requests of any kind.
- `--reference` reports word and character error rates against a reference
  transcript, both as written and with Persian orthography folded away.
- URL inputs: `vatfa <url>` downloads the media first, via the `[url]` extra.
- `vatfa-web`, a local browser interface, via the `[web]` extra. Text appears
  as it decodes, and the page shows the equivalent command line.
- `--no-punctuation`, closing a gap where the option existed in the library
  but nothing on the CLI could reach it.
- `--embed-media`, `--download-dir`.
- Continuous integration across Python 3.9 to 3.13, plus macOS and Windows.

### Changed

- Renamed from `transcribe-fa` to `video-audio-transcriber`, matching the
  repository. Installs two commands: `video-audio-transcriber` and `vatfa`.
- The default language is now `auto` rather than `fa`. Detection runs before
  decoding so the Persian initial prompt can still be chosen from it, and a
  result of Arabic, Urdu or Pashto is flagged as a likely misdetection.
- `json` output reports the source file's name rather than its absolute path,
  which was leaking directory layout into shared files.

### Fixed

- `--help` crashed on Windows with `UnicodeEncodeError`, because the console
  defaults to a code page that cannot encode Persian. Piping a transcript to
  stdout would have failed the same way.

## [0.1.0] - 2026-09-11

Initial release: Whisper transcription with Persian normalization, five output
formats and subtitle cue splitting.
