# video-audio-transcriber

[![CI](https://github.com/webgodo/video-audio-transcriber/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/webgodo/video-audio-transcriber/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Runs 100% offline](https://img.shields.io/badge/runs-100%25%20offline-lightgrey.svg)](#how-it-works)

Offline Persian (Farsi) transcription for audio and video files, built on
[OpenAI Whisper](https://github.com/openai/whisper). Everything runs on your
own machine: no API keys, no uploads, no per-minute fees.

```bash
vatfa lecture.mp3            # -> lecture.txt and lecture.srt next to the file
vatfa interview.mp4 -f srt   # subtitles for a video
```

## How it works

* **Model.** OpenAI's open-source Whisper weights, `large-v3` by default (the
  most accurate Whisper model for Persian). They are run through
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper), a CTranslate2
  port of the same network that is about 4x faster than the reference PyTorch
  code and needs far less memory, with identical accuracy.
* **Input.** Any audio or video container FFmpeg understands (mp3, wav, m4a,
  flac, ogg, opus, mp4, mkv, webm, mov, ...). Video files are decoded
  directly; there is no separate extraction step.
* **Persian-aware clean-up** of the raw model output (see below).
* **Voice activity detection** (Silero VAD, built into faster-whisper) skips
  silence, which is where Whisper tends to hallucinate text.
* **Outputs:** `txt`, `srt`, `vtt`, `json` (optionally with word timestamps)
  and `tsv`.

## Requirements

* Python 3.9 or newer (Linux, macOS, Windows).
* About 3 GB of disk for `large-v3` (downloaded once, cached in
  `~/.cache/huggingface/hub`).
* Optional: an NVIDIA GPU. `large-v3` needs about 4.5 GB of VRAM in fp16 and
  runs 5 to 10x faster than on a CPU. CPU-only works fine; it is just slower,
  and the `medium` or `large-v3-turbo` models are more comfortable there.

## Install

```bash
git clone https://github.com/webgodo/video-audio-transcriber.git
cd video-audio-transcriber
python3 -m venv .venv && source .venv/bin/activate
pip install -e .              # CPU only
pip install -e '.[cuda]'      # NVIDIA GPU: also installs cuBLAS via pip, no system CUDA needed
```

Or as an isolated tool: `pipx install .` (or `pipx install '.[cuda]'`).

The model is downloaded on first use. To fetch it ahead of time:

```bash
vatfa --download-only            # large-v3
vatfa --download-only -m medium
```

Downloads resume if interrupted, so re-run the command if your connection drops.

## Usage

```
vatfa [options] FILE [FILE ...]
```

Inputs can be files or directories (directories are searched recursively for
media files). By default a `.txt` and a `.srt` file are written next to each
input.

```bash
vatfa talk.mp3                             # talk.txt + talk.srt
vatfa talk.mp3 -f txt,json --word-timestamps
vatfa video.mp4 -f srt --max-cue-chars 42  # subtitle-sized cues
vatfa recordings/ -o transcripts/          # batch, outputs in one folder
vatfa call.wav --stdout | grep قرارداد      # plain text on stdout, pipe-friendly
vatfa long-podcast.mp3 --batch-size 8      # GPU: batched decoding, several times faster
vatfa clip.m4a -m medium --device cpu      # lighter model on the CPU
vatfa clip.m4a --task translate -f txt     # English translation instead
vatfa podcast.mp3 -f html                  # interactive page: click a line, it seeks
vatfa --list-models
```

### Interactive transcripts

`-f html` writes a single self-contained page: click any line and the audio
jumps there, words highlight as it plays, and there is a search box.

```bash
vatfa interview.mp3 -f html                 # interview.html, links interview.mp3
vatfa interview.mp3 -f html --embed-media   # one file, audio included
```

The page makes no external requests at all: no CDN, no web font, no
framework. That is deliberate rather than minimalist, because a page that
fetched a font on open would quietly undo this tool's main promise. There is a
test that fails if any absolute URL appears in the output.

Persian and other right-to-left languages are laid out correctly without the
reader configuring anything. `-f html` turns on word timestamps by itself.

One deployment note: when the page links the media rather than embedding it,
seeking needs a server that supports HTTP range requests. GitHub Pages, nginx
and Apache all do. Python's `http.server` does not, so audio opened through it
will play but refuse to seek. `--embed-media` sidesteps the question entirely.

### Measuring accuracy

`--reference` scores a transcript against a known-correct one, so a claim
about quality can be a measurement rather than an assertion.

```bash
vatfa clip.mp3 --reference clip.reference.txt
vatfa recordings/ --reference references/     # matched by file name
```

```
file                              WER      CER     WER*     CER*
----------------------------------------------------------------
clip.mp3                        14.7%     4.7%    15.0%     3.9%
```

The starred columns fold Persian orthography away first: `ی`/`ک`,
Arabic-Indic digits, punctuation and diacritics. `CER*` also ignores every
word separator, so `می‌رود`, `می رود` and `میرود` all score the same.

The difference between the columns is the point. A transcript can score 100%
word error raw and 0% starred, which means the model heard every word
correctly and wrote all of them in the other convention. That gap is what the
clean-up above removes. The starred columns re-segment the text, so they are a
separate measurement rather than the unstarred ones minus something.

### Cleaning up text you already have

The Persian clean-up is useful on its own, so it is also available with no
model, no download and no media file. Point `--text` at a `.txt`, `.srt` or
`.vtt` file, or pipe text through it:

```bash
vatfa --text subtitles.srt                 # cleaned-up copy on stdout
vatfa --text subtitles.srt -o fixed/       # or written to a directory
vatfa --text captions.vtt notes.txt -o out/
echo 'من می روم و كتاب ها را می خوانم?' | vatfa --text -
#   -> من می‌روم و کتاب‌ها را می‌خوانم؟
```

This works on YouTube auto-captions, subtitles downloaded from the web, and
anything a person typed. Subtitle files keep their indices, timestamps, cue
settings, WebVTT headers and `NOTE` blocks byte for byte; only the cue text is
touched, so a malformed file can never come out more broken than it went in.
Legacy windows-1256 files are decoded automatically, which matters because
that encoding cannot represent `ی` or `ک` at all.

### Options

| Option | Meaning |
| --- | --- |
| `-o DIR` | Output directory (default: next to each input). |
| `-f FMT[,FMT...]` | Output formats: `txt`, `srt`, `vtt`, `json`, `tsv`, `html` (default `txt,srt`). |
| `--stdout` | Print the transcript text to stdout as it is produced (no files unless `-f` is given). |
| `--max-cue-chars N` | Split `srt`/`vtt` cues longer than N characters on word boundaries (implies word timestamps). |
| `--embed-media` | For `-f html`: inline the media into the page so the file works on its own. |
| `--rtl-mark` | Prefix subtitle lines with U+200F for players that render trailing `؟` `.` on the wrong side. |
| `--skip-existing` | Skip inputs whose output files already exist (resumable batch runs). |
| `-m MODEL` | `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3` (default), `large-v3-turbo`, or a path / Hugging Face repo of a CTranslate2 Whisper model. |
| `--device auto/cpu/cuda` | Default `auto`: GPU when available, otherwise CPU. |
| `--compute-type TYPE` | Default `float16` on GPU, `int8` on CPU. `int8_float16` halves GPU memory. |
| `--threads N` | CPU threads (default: half the logical cores). |
| `--model-dir DIR`, `--offline` | Model cache location; never download. |
| `-l LANG` | Language (default `fa`). `auto` detects, but see below. |
| `--task translate` | Produce English text instead of Persian. |
| `--beam-size N` | Default 5. `1` is faster and slightly less accurate. |
| `--no-vad` | Disable voice activity detection. |
| `--word-timestamps` | Word-level timing in `json` output. |
| `--prompt TEXT` | Initial prompt (default: a short Persian sentence; `''` disables). |
| `--no-context` | Do not condition on previous text; fixes runaway repetition. |
| `--batch-size N` | Batched decoding (GPU), e.g. `8`. |
| `--no-normalize`, `--no-zwnj`, `--digits MODE` | Persian clean-up controls, see below. |
| `--reference PATH` | Score against a reference transcript: a `.txt` file, or a directory matched by input name. |
| `-q`, `-v` | Quiet / verbose. |

## Persian specifics

**The language is forced to `fa`.** Whisper's language detector regularly
labels Persian speech as Arabic, Urdu or Pashto, after which it transcribes in
the wrong script. Forcing `fa` avoids that. `-l auto` is available for mixed
collections, but expect occasional misdetection on short or noisy clips.

**Initial prompt.** Whisper treats the prompt as text that came before the
audio, so a well-formed Persian sentence with ZWNJs and Persian punctuation
nudges the model to produce the same. This matters: without it, `large-v3`
tends to drop punctuation and ZWNJs entirely (`میخواهم`, `میشه`, `در باره`);
with it, the same audio comes out as `می‌خواهم`, `می‌شه`, `درباره‌ی` with
full punctuation. The default is a neutral sentence;
`--prompt` replaces it (useful for domain words and names that the model
should spell a particular way), `--prompt ''` disables it.

**Text clean-up.** Whisper's Persian output is good but inconsistent in ways
that hurt readability and search. After decoding, each segment is normalised:

| Fix | Example |
| --- | --- |
| Arabic `ي` `ك` `ة` -> Persian `ی` `ک` `ه`; kashida removed | `كتاب` -> `کتاب` |
| Arabic-Indic digits -> Persian digits | `١٤٠٣` -> `۱۴۰۳` |
| ZWNJ between common affixes and stems | `می رود` -> `می‌رود`, `کتاب ها` -> `کتاب‌ها`, `بزرگ ترین` -> `بزرگ‌ترین` |
| Persian punctuation after Persian words | `خوبی?` -> `خوبی؟`, `سلام,` -> `سلام،` |
| Spacing around punctuation and stray ZWNJs | `سلام .خوبی` -> `سلام. خوبی` |

The rules are conservative (for example `می` is not joined before `و`, `که`,
`را`, and thousands separators are left alone). `--no-zwnj` keeps the
character fixes but skips the affix joining; `--no-normalize` returns the raw
model output; `--digits persian` or `--digits western` converts all digits.

**Right-to-left subtitles.** Modern players (VLC, mpv, browsers) render
Persian subtitles correctly. Some older players or editors put a trailing
question mark or full stop on the wrong side of the line; `--rtl-mark` adds an
invisible right-to-left mark at the start of every cue to fix that.

**Subtitle length.** Whisper produces segments of up to about 30 seconds,
which is too long for on-screen subtitles. `--max-cue-chars 42` re-splits
them on word boundaries (also on pauses longer than a second and at 7 seconds
of duration) using word timestamps.

## Choosing a model

| Model | Params | Memory (GPU fp16 / CPU int8) | Persian quality |
| --- | --- | --- | --- |
| `large-v3` | 1550 M | 4.5 GB / 3 GB | best (default) |
| `large-v3-turbo` | 809 M | 2.5 GB / 1.6 GB | close to large-v3, about 4x faster |
| `large-v2` | 1550 M | 4.5 GB / 3 GB | excellent |
| `medium` | 769 M | 2.5 GB / 1.5 GB | good; the CPU sweet spot |
| `small` | 244 M | 1 GB / 0.6 GB | acceptable for clear speech |
| `tiny`, `base` | | | not recommended for Persian |

Measured on a laptop (RTX 4060 8 GB, i9-13900HX) with `large-v3` on a short
clip; longer files get better ratios because model start-up is a fixed cost:

| Setting | Speed | Memory |
| --- | --- | --- |
| GPU, `float16` | ~6x realtime | 4.2 GB VRAM peak (also with `--batch-size 8`) |
| CPU, `int8`, 16 threads | ~1.1x realtime | ~3 GB RAM |

Rules of thumb:

* GPU with 5 GB+ VRAM: `large-v3`. Add `--batch-size 8` for long recordings.
* GPU with less VRAM: `large-v3-turbo`, or `large-v3 --compute-type int8_float16`.
* CPU only: `medium` or `large-v3-turbo`; `large-v3` works but is slow.
  `--beam-size 1` roughly halves the time at a small accuracy cost.

## Troubleshooting

* **Made-up sentences during silence or music** (a classic Whisper failure,
  often a closing phrase like thanks for watching). Keep VAD on (the default)
  and try `--no-context`. Trimming long silent intros also helps.
* **The same phrase repeats over and over.** Use `--no-context`; if it
  persists, `--beam-size 1` or a different model.
* **Output in Arabic script, or in another language.** You probably used
  `-l auto`; drop it so the language is forced to Persian.
* **"GPU initialisation failed ... falling back to CPU".** Install the GPU
  extra: `pip install -e '.[cuda]'`. On a system with CUDA already installed,
  make sure cuBLAS 12 is on the library path.
* **CUDA out of memory.** `--compute-type int8_float16`, `-m large-v3-turbo`,
  a smaller `--batch-size`, or `--device cpu`.
* **Very slow on CPU.** Use `-m medium`, `--beam-size 1`, and check
  `--threads` matches your physical cores.
* **Interrupted model download.** Just run the command again; downloads resume.

## Development

```bash
pip install -e '.[dev]'
pytest
```

The package layout: `transcriber.py` wraps faster-whisper (model loading,
GPU library discovery, decoding), `normalize.py` holds the Persian text
rules, `writers.py` the output formats and subtitle cue splitting, and
`cli.py` the command line.

## License

MIT. Whisper's weights are released by OpenAI under the MIT license as well.
