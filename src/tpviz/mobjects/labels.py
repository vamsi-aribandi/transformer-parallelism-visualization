"""MathTex/Text factories. All LaTeX strings come from tpviz.notation."""

from __future__ import annotations

from manim import MathTex, Text

from tpviz import style
from tpviz.core.tensors import LTensor


def tensor_label(t: LTensor, font_size: float = 22) -> MathTex:
    tex = MathTex(t.tex(), font_size=font_size)
    tex.set_color(style.TEXT_COLOR)
    tex.set_z_index(style.Z_LABEL)
    return tex


def math_label(tex_str: str, font_size: float = 26, color: str = style.TEXT_COLOR) -> MathTex:
    tex = MathTex(tex_str, font_size=font_size)
    tex.set_color(color)
    tex.set_z_index(style.Z_LABEL)
    return tex


def caption_text(s: str, font_size: float = 24, color: str = style.TEXT_COLOR) -> Text:
    txt = Text(s, font_size=font_size, color=color)
    txt.set_z_index(style.Z_LABEL)
    return txt
