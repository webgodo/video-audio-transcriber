"""Word and character error rates against a reference transcript.

Every comparison is reported twice, and the difference is the interesting part:

``raw``
    The text exactly as written. Every orthographic difference is an error:
    ``كتاب`` against ``کتاب``, ``می رود`` against ``می‌رود``, ``١٤٠٣``
    against ``۱۴۰۳``.
``scoring``
    Exactly the conventions ``normalize.py`` fixes are folded away first, so
    only real recognition mistakes count.

A model can therefore score 100% raw word error while scoring 0% in scoring
mode: it heard every word correctly and wrote all of them in the other
convention. That is the case this project's Persian clean-up exists for, and
it can be measured rather than asserted.

The two are separate measurements, not one minus the other. Folding a ZWNJ to
a space changes where the word boundaries are, so the starred rates count
against their own token stream.

There is no new dependency here. Levenshtein over a two-row table is forty
lines, and taking on a package to compute it would undercut a core with three
dependencies.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from .normalize import ZWNJ, normalize_word

log = logging.getLogger("video_audio_transcriber")

#: Above this many tokens per side the O(n*m) table gets slow in Python.
#: Benchmark inputs are short by nature: a hand-checked reference longer than
#: twenty minutes is rare, and Common Voice clips are seconds long. This is a
#: guard rail, not a claim that the implementation scales.
MAX_TOKENS = 3000

_PUNCTUATION = re.compile(r"[^\w\s]|_", re.UNICODE)
_DIACRITICS = re.compile(r"[ً-ْٰـ]")
_WHITESPACE = re.compile(r"\s+")


@dataclass
class ErrorRate:
    """Counts from one alignment, and the rate they imply."""

    hits: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    @property
    def reference_length(self) -> int:
        return self.hits + self.substitutions + self.deletions

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        """Errors per reference token. Zero for an empty reference, not a crash."""
        return self.errors / self.reference_length if self.reference_length else 0.0

    def __add__(self, other: "ErrorRate") -> "ErrorRate":
        return ErrorRate(
            self.hits + other.hits,
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
        )

    def __str__(self) -> str:
        return (
            f"{self.rate:.1%} ({self.substitutions}S {self.deletions}D {self.insertions}I "
            f"of {self.reference_length})"
        )


def fold(text: str, *, drop_separators: bool = False) -> str:
    """Reduce text to what was said, dropping how it was spelled.

    Applies the character fixes (Arabic yeh and kaf to Persian, Arabic-Indic
    to Western digits), turns a ZWNJ into a space so ``می‌رود`` and ``می رود``
    agree, and removes punctuation and Arabic diacritics.

    ``drop_separators`` additionally removes every space, which is the only
    way to make ``می‌رود``, ``می رود`` and ``میرود`` all equal. Persian word
    boundaries are exactly what the two spellings disagree about, so no
    word-level comparison can treat all three as the same; a character-level
    one can.
    """
    text = normalize_word(unicodedata.normalize("NFC", text), digits="western")
    text = text.replace(ZWNJ, " ")
    text = _DIACRITICS.sub("", text)
    text = _PUNCTUATION.sub(" ", text)
    text = _WHITESPACE.sub("" if drop_separators else " ", text)
    return text.strip()


def tokenize(text: str, *, scoring: bool = False) -> List[str]:
    """Split into words, optionally folding orthography away first."""
    return (fold(text) if scoring else text).split()


def error_rate(reference: Sequence[str], hypothesis: Sequence[str]) -> ErrorRate:
    """Levenshtein alignment of two token sequences, keeping the operation counts.

    Two rows of state, each cell carrying the counts accumulated along its own
    best path, so no backtrace matrix is needed.
    """
    ref, hyp = list(reference), list(hypothesis)
    if len(ref) > MAX_TOKENS or len(hyp) > MAX_TOKENS:
        raise ValueError(
            f"too long to align exactly: {len(ref)} vs {len(hyp)} tokens, limit {MAX_TOKENS}. "
            "Score shorter excerpts instead."
        )
    if not ref:
        return ErrorRate(insertions=len(hyp))
    if not hyp:
        return ErrorRate(deletions=len(ref))

    # Cell: (cost, hits, substitutions, deletions, insertions)
    previous = [(j, 0, 0, 0, j) for j in range(len(hyp) + 1)]
    for i, want in enumerate(ref, 1):
        current = [(i, 0, 0, i, 0)]
        for j, got in enumerate(hyp, 1):
            same = want == got
            d = previous[j - 1]
            substitute = (
                d[0] + (0 if same else 1),
                d[1] + (1 if same else 0),
                d[2] + (0 if same else 1),
                d[3],
                d[4],
            )
            u = previous[j]  # a reference token with nothing aligned to it
            delete = (u[0] + 1, u[1], u[2], u[3] + 1, u[4])
            left = current[j - 1]  # a hypothesis token with nothing to align to
            insert = (left[0] + 1, left[1], left[2], left[3], left[4] + 1)
            current.append(min(substitute, delete, insert))
        previous = current

    _, hits, substitutions, deletions, insertions = previous[-1]
    return ErrorRate(hits, substitutions, deletions, insertions)


def word_error_rate(reference: str, hypothesis: str, *, scoring: bool = False) -> ErrorRate:
    return error_rate(tokenize(reference, scoring=scoring), tokenize(hypothesis, scoring=scoring))


def char_error_rate(reference: str, hypothesis: str, *, scoring: bool = False) -> ErrorRate:
    """Character error rate. In scoring mode every separator is removed first,
    so ZWNJ, space and joined spellings of the same word all score identical."""
    if scoring:
        ref, hyp = fold(reference, drop_separators=True), fold(hypothesis, drop_separators=True)
    else:
        ref, hyp = " ".join(reference.split()), " ".join(hypothesis.split())
    return error_rate(list(ref), list(hyp))


def compare(reference: str, hypothesis: str) -> Dict[str, ErrorRate]:
    """Word and character rates, before and after folding orthography away."""
    return {
        "wer": word_error_rate(reference, hypothesis),
        "cer": char_error_rate(reference, hypothesis),
        "wer_scoring": word_error_rate(reference, hypothesis, scoring=True),
        "cer_scoring": char_error_rate(reference, hypothesis, scoring=True),
    }


def read_reference(media: Path, where: Union[str, Path]) -> Optional[str]:
    """Find the reference text for one input: a file, or a directory to search.

    In a directory, ``clip.mp3`` matches ``clip.txt`` or ``clip.reference.txt``.
    """
    from .textfix import read_text

    where = Path(where)
    if where.is_file():
        return read_text(where)
    if where.is_dir():
        for name in (f"{media.stem}.reference.txt", f"{media.stem}.txt"):
            candidate = where / name
            if candidate.is_file():
                return read_text(candidate)
        log.warning("no reference for %s in %s", media.name, where)
        return None
    log.warning("reference not found: %s", where)
    return None


def format_table(rows: Sequence) -> str:
    """Render per-file rates plus a total, for the terminal or a README."""
    header = f"{'file':<28}{'WER':>9}{'CER':>9}{'WER*':>9}{'CER*':>9}"
    lines = [header, "-" * len(header)]
    for name, rates in rows:
        label = name if len(name) <= 27 else "..." + name[-24:]
        lines.append(
            f"{label:<28}{rates['wer'].rate:>8.1%}{rates['cer'].rate:>9.1%}"
            f"{rates['wer_scoring'].rate:>9.1%}{rates['cer_scoring'].rate:>9.1%}"
        )
    if len(rows) > 1:
        total = {k: ErrorRate() for k in rows[0][1]}
        for _, rates in rows:
            for key, value in rates.items():
                total[key] = total[key] + value
        lines.append("-" * len(header))
        lines.append(
            f"{'total':<28}{total['wer'].rate:>8.1%}{total['cer'].rate:>9.1%}"
            f"{total['wer_scoring'].rate:>9.1%}{total['cer_scoring'].rate:>9.1%}"
        )
    lines.append("")
    lines.append("* Persian orthography folded away: ی/ک, Arabic-Indic digits, punctuation,")
    lines.append("  diacritics. CER* also ignores every word separator, so می‌رود, می رود and")
    lines.append("  میرود score the same. Starred columns re-segment the text and are therefore")
    lines.append("  a separate measurement, not the unstarred ones minus something.")
    return "\n".join(lines)
