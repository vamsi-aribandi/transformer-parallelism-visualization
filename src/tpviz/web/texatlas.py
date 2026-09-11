"""LaTeX -> shared glyph atlas for the web player.

Every equation is compiled once via manim's TinyTeX/dvisvgm pipeline (the same
one the videos use, so notation is pixel-identical), then decomposed:
  - glyph <path> outlines are deduped GLOBALLY by their path data into an
    atlas (id "G0", "G1", ...) — dvisvgm's per-document ids like `g1-66` are
    NOT stable across runs, so dedupe keys on the `d` attribute;
  - each equation keeps only a list of <use> placements (glyph id, x, y) and
    any <rect> rules (fraction bars), plus its viewBox.
Rendered client-side with fill=currentColor, so equations recolor by state.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from manim import config
from manim.utils.tex_file_writing import tex_to_svg_file

_PATH_RE = re.compile(r"<path id='([^']+)' d='([^']+)'/>")
_USE_RE = re.compile(r"<use x='([-\d.]+)' y='([-\d.]+)' xlink:href='#([^']+)'/>")
_RECT_RE = re.compile(
    r"<rect x='([-\d.]+)' y='([-\d.]+)' height='([-\d.]+)' width='([-\d.]+)'"
)
_VIEWBOX_RE = re.compile(r"viewBox='([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+)'")


def eq_key(tex: str) -> str:
    return hashlib.sha1(tex.encode()).hexdigest()[:8]


@dataclass
class TexAtlas:
    glyphs: dict[str, str] = field(default_factory=dict)  # atlas id -> path data
    _by_d: dict[str, str] = field(default_factory=dict)  # path data -> atlas id
    eqs: dict[str, dict] = field(default_factory=dict)  # eq key -> layout
    tex_of: dict[str, str] = field(default_factory=dict)  # eq key -> source tex

    def _glyph_id(self, d: str) -> str:
        gid = self._by_d.get(d)
        if gid is None:
            gid = f"G{len(self.glyphs)}"
            self._by_d[d] = gid
            self.glyphs[gid] = d
        return gid

    def add(self, tex: str) -> str:
        """Compile (or reuse) `tex`; returns its equation key."""
        key = eq_key(tex)
        if key in self.eqs:
            return key
        svg_path = tex_to_svg_file(
            tex, environment="align*", tex_template=config.tex_template
        )
        svg = svg_path.read_text()
        vb = _VIEWBOX_RE.search(svg)
        x0, y0, w, h = (float(v) for v in vb.groups())
        local = {m.group(1): self._glyph_id(m.group(2)) for m in _PATH_RE.finditer(svg)}
        uses = [
            [local[m.group(3)], round(float(m.group(1)) - x0, 2), round(float(m.group(2)) - y0, 2)]
            for m in _USE_RE.finditer(svg)
        ]
        rects = [
            [
                round(float(m.group(1)) - x0, 2),
                round(float(m.group(2)) - y0, 2),
                round(float(m.group(4)), 2),  # width
                round(float(m.group(3)), 2),  # height
            ]
            for m in _RECT_RE.finditer(svg)
        ]
        self.eqs[key] = {"w": round(w, 2), "h": round(h, 2), "u": uses, "r": rects}
        self.tex_of[key] = tex
        return key

    def emit(self) -> dict:
        return {"glyphs": self.glyphs, "eq": self.eqs}
