"""Persian (Farsi) text clean-up for Whisper output.

Whisper's Persian output is generally good, but it is inconsistent about a
few things that matter for readable, searchable Persian text:

* Arabic ``ي`` / ``ك`` instead of Persian ``ی`` / ``ک`` (they look alike but
  are different code points, which breaks search and sorting).
* Arabic-Indic digits (``٠١٢``) instead of Persian digits (``۰۱۲``).
* A space instead of a zero-width non-joiner (ZWNJ, U+200C) between common
  affixes and their stems: ``می رود`` -> ``می‌رود``, ``کتاب ها`` -> ``کتاب‌ها``.
* Latin ``?`` ``,`` ``;`` after Persian words instead of ``؟`` ``،`` ``؛``.
* Stray spaces before punctuation, doubled spaces, ZWNJ next to spaces.

All rules are deliberately conservative: they only touch constructs that are
unambiguous in ordinary Persian prose, so they are safe to run by default.
Use ``--no-normalize`` on the CLI to get the raw model output.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .transcriber import Transcript

ZWNJ = "\u200c"
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
WESTERN_DIGITS = "0123456789"

DIGIT_MODES = ("keep", "persian", "western")

# ---------------------------------------------------------------- characters

_CHAR_MAP: dict = {
    "\u064a": "\u06cc",  # ي ARABIC YEH          -> ی FARSI YEH
    "\u0649": "\u06cc",  # ى ALEF MAKSURA        -> ی
    "\u06d2": "\u06cc",  # ے YEH BARREE (Urdu)   -> ی
    "\u0643": "\u06a9",  # ك ARABIC KAF          -> ک KEHEH
    "\u06aa": "\u06a9",  # ڪ SWASH KAF           -> ک
    "\u0629": "\u0647",  # ة TEH MARBUTA         -> ه
    "\u06c1": "\u0647",  # ہ HEH GOAL (Urdu)     -> ه
    "\u0640": "",        # ـ TATWEEL / kashida   -> (removed)
    "\u00a0": " ",       # no-break space        -> space
    "\u200b": "",        # zero-width space      -> (removed)
    "\ufeff": "",        # BOM / ZWNBSP          -> (removed)
}
_CHAR_MAP.update(dict(zip(ARABIC_INDIC_DIGITS, PERSIAN_DIGITS)))
_CHAR_TABLE = str.maketrans(_CHAR_MAP)
_TO_PERSIAN_DIGITS = str.maketrans(WESTERN_DIGITS, PERSIAN_DIGITS)
_TO_WESTERN_DIGITS = str.maketrans(PERSIAN_DIGITS + ARABIC_INDIC_DIGITS, WESTERN_DIGITS * 2)

# One Persian/Arabic-script letter (no digits, diacritics or punctuation).
_L = "[\u0621-\u063a\u0641-\u064a\u066e-\u06d3\u06d5\u06ee\u06ef\u06fa-\u06fc\u06ff]"
# One letter or Persian digit: "Persian context" for the punctuation rules.
_LD = "[\u0621-\u063a\u0641-\u064a\u066e-\u06d3\u06d5\u06ee\u06ef\u06f0-\u06fc\u06ff]"
_L_RE = re.compile(_L)

# ------------------------------------------------------------------- affixes

# If "می" is followed by one of these, it is probably not the verb prefix
# (e.g. the noun "می" = wine), so it is left alone.
_MI_NO_JOIN = frozenset(
    "و که را به از در با تا بر هم یا این آن چه هر من تو او ما شما".split()
)
_MI_RE = re.compile(rf"(?<!{_L})(?<!{ZWNJ})(ن?می) (?=({_L}+))")
_PLURAL_RE = re.compile(
    rf"(?<={_L}) (هایشان|هایتان|هایمان|هایی|هایم|هایت|هایش|های|ها)(?!{_L})"
)
_SUPERLATIVE_RE = re.compile(rf"(?<={_L}) (ترین)(?!{_L})")


def _join_mi(match: "re.Match[str]") -> str:
    if match.group(2) in _MI_NO_JOIN:
        return match.group(0)
    return match.group(1) + ZWNJ


def fix_affixes(text: str) -> str:
    """Replace the space between common affixes and their stem with a ZWNJ."""
    text = _MI_RE.sub(_join_mi, text)
    text = _PLURAL_RE.sub(lambda m: ZWNJ + m.group(1), text)
    text = _SUPERLATIVE_RE.sub(lambda m: ZWNJ + m.group(1), text)
    return text


# --------------------------------------------------------------- punctuation

_LATIN_PUNCT = (
    (re.compile(rf"(?<={_LD})[ \t]*\?"), "؟"),
    (re.compile(rf"(?<={_LD})[ \t]*,(?![0-9۰-۹])"), "،"),
    (re.compile(rf"(?<={_LD})[ \t]*;"), "؛"),
)
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+(?=[،؛؟!.:»)\]])")
_SPACE_AFTER_OPEN = re.compile(r"(?<=[«(\[])[ \t]+")
_CLOSERS = frozenset("»)]}،؛؟!.:…\"'")
_DIGITS = frozenset(WESTERN_DIGITS + PERSIAN_DIGITS)


def _space_after_punct(text: str) -> str:
    out = []
    last = len(text) - 1
    for i, ch in enumerate(text):
        out.append(ch)
        if i == last:
            break
        nxt = text[i + 1]
        if ch in "،؛؟!":
            if nxt.isspace() or nxt in _CLOSERS:
                continue
            if ch == "،" and i > 0 and text[i - 1] in _DIGITS and nxt in _DIGITS:
                continue  # thousands separator: ۱،۰۰۰
            out.append(" ")
        elif ch in ".:" and _L_RE.match(nxt):
            out.append(" ")  # "سلام.خوبی" -> "سلام. خوبی"; leaves 3.5 and 12:30 alone
    return "".join(out)


def fix_punctuation(text: str) -> str:
    """Use Persian punctuation after Persian words and tidy spacing around it."""
    for pattern, replacement in _LATIN_PUNCT:
        text = pattern.sub(replacement, text)
    text = _SPACE_BEFORE_PUNCT.sub("", text)
    text = _SPACE_AFTER_OPEN.sub("", text)
    return _space_after_punct(text)


# ---------------------------------------------------------------- whitespace

_ZWNJ_RUN = re.compile(rf"{ZWNJ}{{2,}}")
# A ZWNJ only means something between two letters; drop it anywhere else.
_STRAY_ZWNJ = re.compile(rf"(?<!{_L}){ZWNJ}+|{ZWNJ}+(?!{_L})")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def _convert_digits(text: str, digits: str) -> str:
    if digits == "persian":
        return text.translate(_TO_PERSIAN_DIGITS)
    if digits == "western":
        return text.translate(_TO_WESTERN_DIGITS)
    if digits != "keep":
        raise ValueError(f"unknown digits mode {digits!r}; expected one of {DIGIT_MODES}")
    return text


def normalize_word(text: str, *, digits: str = "keep") -> str:
    """Character-level fixes only (safe for single words / tokens)."""
    text = unicodedata.normalize("NFC", text).translate(_CHAR_TABLE)
    return _convert_digits(text, digits)


def normalize(
    text: str,
    *,
    digits: str = "keep",
    zwnj: bool = True,
    punctuation: bool = True,
) -> str:
    """Clean up a Persian sentence or paragraph produced by Whisper.

    ``digits``: ``keep`` (only fix Arabic-Indic digits), ``persian`` or ``western``.
    ``zwnj``: join common affixes (می‌، ‌ها، ‌ترین) with a ZWNJ.
    ``punctuation``: use ؟ ، ؛ after Persian words and fix spacing.
    """
    if not text:
        return text
    text = normalize_word(text, digits=digits)
    text = _ZWNJ_RUN.sub(ZWNJ, text)
    text = _STRAY_ZWNJ.sub("", text)
    text = _MULTI_SPACE.sub(" ", text)
    if punctuation:
        text = fix_punctuation(text)
    if zwnj:
        text = fix_affixes(text)
    text = _MULTI_SPACE.sub(" ", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def normalize_transcript(
    transcript: "Transcript",
    *,
    digits: str = "keep",
    zwnj: bool = True,
    punctuation: bool = True,
) -> "Transcript":
    """Normalise every segment (and word) of a transcript in place."""
    for segment in transcript.segments:
        segment.text = normalize(segment.text, digits=digits, zwnj=zwnj, punctuation=punctuation)
        for word in segment.words:
            word.word = normalize_word(word.word, digits=digits)
    return transcript
