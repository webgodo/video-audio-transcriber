"""Render a transcript as a single self-contained, interactive HTML page.

Click any line to seek the audio, watch the words highlight as it plays, and
search the whole transcript. The result is one file with no external requests
of any kind: no CDN, no web font, no framework. That is not an aesthetic
choice. This project's claim is that nothing leaves your machine, and a page
that phoned a font CDN on open would quietly make that false.

Right-to-left is applied for Persian and the other RTL languages Whisper can
produce, so the page reads correctly without the viewer configuring anything.
"""

from __future__ import annotations

import base64
import html
import json
import mimetypes
import os
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:  # pragma: no cover
    from .transcriber import Transcript

#: Whisper languages written right to left.
RTL_LANGUAGES = frozenset({"fa", "ar", "he", "ur", "ps", "sd", "yi"})
#: Above this, a data: URI makes the page unpleasant to open.
EMBED_WARN_BYTES = 50 * 1024 * 1024

_PAGE = Template(
    """<!doctype html>
<html lang="$lang" dir="$dir">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title</title>
<style>
:root {
  --bg: #fbfbfa; --fg: #1c1c1a; --muted: #6b6b66; --line: #e4e4e0;
  --accent: #1f6feb; --hit: #ffe680; --active: #eaf1fd; --card: #fff;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #16171a; --fg: #e9e9e6; --muted: #9a9a94; --line: #2c2e33;
    --accent: #6aa8ff; --hit: #7a6420; --active: #1e2836; --card: #1c1e22;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font-family: "Vazirmatn", "Noto Naskh Arabic", "Segoe UI", Tahoma, system-ui, sans-serif;
  line-height: 1.9; font-size: 17px;
}
.wrap { max-width: 820px; margin: 0 auto; padding: 24px 20px 80px; }
header h1 { font-size: 1.35rem; margin: 0 0 4px; font-weight: 650; }
.meta { color: var(--muted); font-size: .82rem; letter-spacing: .01em; }
.meta span + span::before { content: " · "; }
.bar {
  position: sticky; top: 0; z-index: 5; background: var(--bg);
  padding: 14px 0 10px; border-bottom: 1px solid var(--line); margin-bottom: 8px;
}
audio, video { width: 100%; display: block; }
video { max-height: 380px; background: #000; border-radius: 8px; }
.tools { display: flex; gap: 8px; align-items: center; margin-top: 10px; }
input[type=search] {
  flex: 1; padding: 8px 12px; font: inherit; font-size: .92rem; color: var(--fg);
  background: var(--card); border: 1px solid var(--line); border-radius: 8px;
}
input[type=search]:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
#count { color: var(--muted); font-size: .82rem; white-space: nowrap; }
.seg {
  display: flex; gap: 12px; align-items: baseline; padding: 7px 10px;
  border-radius: 8px; margin: 1px 0; cursor: pointer;
}
.seg:hover { background: var(--card); }
.seg.on { background: var(--active); }
.seg.hidden { display: none; }
.t {
  flex: none; color: var(--muted); font-size: .76rem; font-variant-numeric: tabular-nums;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; padding-top: .28em;
  direction: ltr; unicode-bidi: isolate; min-width: 4.2em;
}
.txt { flex: 1; }
.w { border-radius: 3px; padding: 0 1px; }
.w.on { background: var(--accent); color: #fff; }
mark { background: var(--hit); color: inherit; border-radius: 3px; }
footer { margin-top: 36px; color: var(--muted); font-size: .78rem; text-align: center; }
footer a { color: var(--accent); }
.empty { color: var(--muted); padding: 24px 10px; }
@media print { .bar { position: static; } .seg { break-inside: avoid; } }
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>$title</h1>
  <div class="meta">$meta</div>
</header>

<div class="bar">
  $media
  <div class="tools">
    <input type="search" id="q" placeholder="$search_label" autocomplete="off" spellcheck="false">
    <span id="count"></span>
  </div>
</div>

<main id="out"></main>

<footer>$footer</footer>
</div>

<script type="application/json" id="data">$payload</script>
<script>
(function () {
  "use strict";
  var data = JSON.parse(document.getElementById("data").textContent);
  var segments = data.segments || [];
  var media = document.querySelector("audio, video");
  var out = document.getElementById("out");
  var q = document.getElementById("q");
  var count = document.getElementById("count");
  var rows = [];

  function clock(s) {
    s = Math.max(0, s || 0);
    var m = Math.floor(s / 60), sec = Math.floor(s % 60);
    var h = Math.floor(m / 60);
    m = m % 60;
    var pad = function (n) { return (n < 10 ? "0" : "") + n; };
    return (h ? h + ":" + pad(m) : m) + ":" + pad(sec);
  }

  segments.forEach(function (seg) {
    var row = document.createElement("div");
    row.className = "seg";
    var t = document.createElement("span");
    t.className = "t";
    t.textContent = clock(seg.start);
    var txt = document.createElement("span");
    txt.className = "txt";
    var words = seg.words || [];
    if (words.length) {
      words.forEach(function (w, i) {
        var el = document.createElement("span");
        el.className = "w";
        el.textContent = (i ? " " : "") + w.word;
        el.dataset.start = w.start;
        el.dataset.end = w.end;
        txt.appendChild(el);
      });
    } else {
      txt.textContent = seg.text;
    }
    row.appendChild(t);
    row.appendChild(txt);
    row.addEventListener("click", function () { seek(seg.start); });
    out.appendChild(row);
    rows.push({ el: row, txt: txt, seg: seg, words: words, plain: seg.text });
  });

  if (!segments.length) {
    out.innerHTML = '<p class="empty">No speech was transcribed.</p>';
  }

  function seek(time) {
    if (!media) return;
    media.currentTime = time;
    if (media.paused) { media.play().catch(function () {}); }
  }

  var lastRow = null, lastWord = null;
  function track() {
    if (!media) return;
    var now = media.currentTime;
    var hit = null;
    for (var i = 0; i < rows.length; i++) {
      if (now >= rows[i].seg.start && now <= rows[i].seg.end) { hit = rows[i]; break; }
    }
    if (hit !== lastRow) {
      if (lastRow) lastRow.el.classList.remove("on");
      if (hit) {
        hit.el.classList.add("on");
        var box = hit.el.getBoundingClientRect();
        if (box.top < 90 || box.bottom > window.innerHeight - 20) {
          hit.el.scrollIntoView({ block: "center", behavior: "smooth" });
        }
      }
      lastRow = hit;
    }
    var word = null;
    if (hit && hit.words.length) {
      var spans = hit.txt.children;
      for (var j = 0; j < spans.length; j++) {
        var s = parseFloat(spans[j].dataset.start), e = parseFloat(spans[j].dataset.end);
        if (now >= s && now <= e) { word = spans[j]; break; }
      }
    }
    if (word !== lastWord) {
      if (lastWord) lastWord.classList.remove("on");
      if (word) word.classList.add("on");
      lastWord = word;
    }
  }
  if (media) media.addEventListener("timeupdate", track);

  function escapeRe(s) { return s.replace(/[.*+?^$$()|[\\]\\\\{}]/g, "\\\\$$&"); }

  function search() {
    var term = q.value.trim();
    if (!term) {
      rows.forEach(function (r) {
        r.el.classList.remove("hidden");
        if (r.words.length) {
          var spans = r.txt.children;
          for (var k = 0; k < spans.length; k++) {
            spans[k].innerHTML = spans[k].textContent;
          }
        } else {
          r.txt.textContent = r.plain;
        }
      });
      count.textContent = "";
      return;
    }
    var re = new RegExp(escapeRe(term), "gi");
    var found = 0;
    rows.forEach(function (r) {
      var match = re.test(r.plain);
      re.lastIndex = 0;
      r.el.classList.toggle("hidden", !match);
      if (match) found++;
      if (r.words.length) {
        var spans = r.txt.children;
        for (var k = 0; k < spans.length; k++) {
          var raw = spans[k].textContent;
          spans[k].innerHTML = re.test(raw)
            ? raw.replace(re, function (m) { return "<mark>" + m + "</mark>"; })
            : raw;
          re.lastIndex = 0;
        }
      } else {
        r.txt.innerHTML = match
          ? r.plain.replace(re, function (m) { return "<mark>" + m + "</mark>"; })
          : r.plain;
      }
    });
    count.textContent = found + "/" + rows.length;
  }
  q.addEventListener("input", search);

  document.addEventListener("keydown", function (e) {
    if (e.key === "/" && document.activeElement !== q) { e.preventDefault(); q.focus(); }
    if (e.key === "Escape" && document.activeElement === q) { q.value = ""; search(); q.blur(); }
    if (e.key === " " && document.activeElement !== q && media) {
      e.preventDefault();
      media.paused ? media.play().catch(function () {}) : media.pause();
    }
  });
})();
</script>
</body>
</html>
"""
)


def _escape_payload(data: dict) -> str:
    """JSON for a ``<script>`` block.

    ``</script>`` anywhere in the transcript would close the tag early and
    break the page, so the sequence is escaped. ``\\u003c`` is valid JSON and
    parses back to the original text.
    """
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return text.replace("</", "<\\/").replace("<!--", "\\u003c!--")


def _data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _media_tag(media: Optional[Path], dest: Optional[Path], embed: bool) -> str:
    if media is None:
        return ""
    mime = mimetypes.guess_type(media.name)[0] or ""
    tag = "video" if mime.startswith("video/") else "audio"
    if embed and media.is_file():
        src = _data_uri(media)
    else:
        src = media.name
        if dest is not None:
            try:
                src = os.path.relpath(media, dest.parent).replace(os.sep, "/")
            except ValueError:  # different drive on Windows
                src = media.as_uri()
    return f'<{tag} controls preload="metadata" src="{html.escape(src, quote=True)}"></{tag}>'


def render_html(
    transcript: "Transcript",
    *,
    media: Optional[Union[str, Path]] = None,
    dest: Optional[Union[str, Path]] = None,
    embed_media: bool = False,
    title: str = "",
) -> str:
    """Render ``transcript`` as one self-contained HTML page.

    ``media`` is the audio or video the transcript came from, linked relative
    to ``dest`` (the path the page is being written to) unless ``embed_media``
    inlines it as a data URI.
    """
    from .writers import to_dict

    media_path = Path(media) if media else None
    dest_path = Path(dest) if dest else None
    rtl = transcript.language in RTL_LANGUAGES
    name = title or (media_path.name if media_path else Path(transcript.source).name) or "transcript"

    data = to_dict(transcript)
    minutes, seconds = divmod(int(round(transcript.duration)), 60)
    meta = [
        f"<span>{html.escape(transcript.language)}</span>",
        f"<span>{minutes}:{seconds:02d}</span>",
        f"<span>{len(transcript.segments)} segments</span>",
    ]
    if transcript.model:
        meta.insert(0, f"<span>{html.escape(transcript.model)}</span>")

    return _PAGE.substitute(
        lang=html.escape(transcript.language or "fa", quote=True),
        dir="rtl" if rtl else "ltr",
        title=html.escape(name),
        meta="".join(meta),
        media=_media_tag(media_path, dest_path, embed_media),
        search_label="جست‌وجو…" if rtl else "Search…",
        # Persian on an RTL page: an English sentence inside a dir="rtl" block
        # has its trailing full stop reordered to the left, which looks broken.
        footer=(
            "روی هر خط بزنید تا صدا به همان‌جا برود. کلید / برای جست‌وجو، فاصله برای پخش و مکث."
            if rtl
            else "Click a line to jump there. Press / to search, space to play or pause."
        ),
        payload=_escape_payload(data),
    )
