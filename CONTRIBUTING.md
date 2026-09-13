# Contributing

Bug reports and pull requests are welcome.

## Getting set up

```bash
git clone https://github.com/webgodo/video-audio-transcriber
cd video-audio-transcriber
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest && ruff check .
```

Add `.[cuda]` for GPU inference, `.[web]` for the browser interface, `.[url]`
for downloading from URLs.

## The one rule that matters

**Tests must not need a model download or a GPU.**

Every heavy import in this package happens inside a function, so the whole
suite runs with `pip install --no-deps -e .` plus pytest, in well under a
second, on every supported Python. That is what lets CI cover Python 3.9
through 3.13 plus macOS and Windows in about a minute, and it is worth
protecting.

If you need to exercise the model, test the code around it against a fake:
`tests/test_pipeline.py` has one, and `tests/test_fetch.py` stubs the
downloader the same way. No test may touch the network.

## Style

- `ruff check .` must pass. The configuration lives in `pyproject.toml` and CI
  pins the same version. Do not run `ruff format`: some literals are
  column-aligned on purpose and the formatter wrecks them.
- Plain pytest functions, no fixture indirection unless it earns its place.
- Comments explain why, not what. If a line looks odd and is correct anyway,
  say why it is there.

## Persian specifics

`normalize.py` is deliberately conservative. A rule belongs there only if it
is unambiguous in ordinary Persian prose, because it runs by default on
everyone's transcripts. Anything arguable belongs behind a flag.

If you add or change a rule, add a test with a real example, and check
`--text` round-trips a subtitle file without touching timestamps.
