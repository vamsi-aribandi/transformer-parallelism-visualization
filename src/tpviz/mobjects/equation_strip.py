"""Persistent bottom strip showing the current operation in book notation."""

from __future__ import annotations

from manim import DOWN, Animation, FadeTransform, MathTex, Rectangle, VGroup, config

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
        self.content: MathTex | None = None
        self.add(band)

    def show(self, tex_str: str, *, color: str = style.TEXT_COLOR) -> Animation:
        new = math_label(tex_str, font_size=30, color=color)
        new.set_z_index(style.Z_STRIP + 1)
        max_w = self.band.width - 1.0
        if new.width > max_w:
            new.scale(max_w / new.width)
        new.move_to(self.band.get_center())
        if self.content is None:
            self.content = new
            self.add(new)
            from manim import FadeIn

            return FadeIn(new)
        old, self.content = self.content, new
        self.remove(old)
        self.add(new)
        return FadeTransform(old, new)
