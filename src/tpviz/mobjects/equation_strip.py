"""Persistent bottom strip showing the current operation in book notation.

During the backward pass, `note` carries the forward equation the current
backward step was derived from — shown as a smaller muted line above the op.
"""

from __future__ import annotations

from manim import DOWN, Animation, FadeIn, FadeTransform, Rectangle, VGroup, config

from tpviz import style
from tpviz.mobjects.labels import math_label


class EquationStrip(VGroup):
    def __init__(self, height: float = 0.9):
        super().__init__()
        band = Rectangle(width=config.frame_width, height=height, stroke_width=0)
        band.set_fill("#151b24", opacity=0.95)
        band.to_edge(DOWN, buff=0.0)
        band.set_z_index(style.Z_STRIP)
        self.band = band
        self.content: VGroup | None = None
        self.add(band)

    def show(self, tex_str: str, *, color: str = style.TEXT_COLOR,
             note: str | None = None) -> Animation:
        main = math_label(tex_str, font_size=30, color=color)
        main.set_z_index(style.Z_STRIP + 1)
        max_w = self.band.width - 1.0
        if main.width > max_w:
            main.scale(max_w / main.width)
        if note is not None:
            note_m = math_label(
                rf"\text{{from forward:}}\quad {note}", font_size=19, color=style.MUTED_TEXT
            )
            note_m.set_z_index(style.Z_STRIP + 1)
            if note_m.width > max_w:
                note_m.scale(max_w / note_m.width)
            new = VGroup(note_m, main).arrange(DOWN, buff=0.10)
        else:
            new = VGroup(main)
        if new.height > self.band.height - 0.12:
            new.scale((self.band.height - 0.12) / new.height)
        new.move_to(self.band.get_center())
        if self.content is None:
            self.content = new
            self.add(new)
            return FadeIn(new)
        old, self.content = self.content, new
        self.remove(old)
        self.add(new)
        return FadeTransform(old, new)
