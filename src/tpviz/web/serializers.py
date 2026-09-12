"""Mobject subtrees -> semantic units the JS renderer can redraw.

`iter_units` walks a scene's top-level mobjects and yields the semantic leaves
(a whole ActivationDeck is ONE unit — its slab/ghost internals are a pure
function of (tensor, mesh, device) redrawn client-side; a DashedVMobject is
one unit, not forty dashes). `serialize` produces the spawn-time draw spec;
`state` produces the per-sample animatable state {x, y, w, o}.

Unknown mobject types are collected in UNSERIALIZED so the exporter can fail
loudly instead of silently dropping visuals.
"""

from __future__ import annotations

from typing import Iterator

from manim import (
    Arc,
    DashedVMobject,
    Dot,
    Line,
    MathTex,
    Mobject,
    Rectangle,
    RoundedRectangle,
    Square,
    Text,
    VMobject,
)

from tpviz.mobjects.tensor_mobject import ActivationDeck, TensorVis, WeightRect

UNSERIALIZED: list[str] = []

_UNIT_TYPES = (TensorVis, MathTex, Text, DashedVMobject, Square, Dot, Arc, Line, Rectangle)


def _hex(c) -> str:
    try:
        return c.to_hex()
    except AttributeError:
        return str(c)


def iter_units(m: Mobject) -> Iterator[Mobject]:
    if getattr(m, "web_role", None) in ("counter", "chrome"):
        return
    if isinstance(m, _UNIT_TYPES):
        yield m
        return
    subs = list(m.submobjects)
    if subs:
        for s in subs:
            yield from iter_units(s)
    elif isinstance(m, VMobject) and m.has_points():
        yield m  # unknown leaf: serialize() will register the gap


def _opacity(m: Mobject) -> float:
    best = 0.0
    for sm in m.family_members_with_points():
        best = max(best, float(sm.fill_opacity or 0), float(sm.stroke_opacity or 0))
    return round(min(best, 1.0), 3)


def state(m: Mobject) -> dict:
    c = m.get_center()
    return {
        "x": round(float(c[0]), 4),
        "y": round(float(c[1]), 4),
        "w": round(float(m.width), 4) if m.width else 0.0,
        "o": _opacity(m),
    }


def _tensor_fields(m: TensorVis) -> dict:
    t = m.t
    return {
        "tensor": {
            "name": t.name,
            "dims": list(t.dims),
            "shard": dict(t.sharding),
            "partial": sorted(t.partial),
            "kind": t.kind,
            "device": m.device,
            "sharded": m.sharded,
            "mesh": dict(m.mesh.axes),
        },
        "tex": t.tex(),
    }


def serialize(m: Mobject) -> dict | None:
    spec: dict | None = None
    if isinstance(m, ActivationDeck):
        spec = {"c": "deck", **_tensor_fields(m)}
    elif isinstance(m, WeightRect):
        spec = {"c": "wrect", **_tensor_fields(m)}
    elif isinstance(m, MathTex):
        spec = {"c": "eq", "tex": m.tex_string, "color": _hex(m.get_color())}
    elif isinstance(m, Text):
        fam = m.family_members_with_points()
        color = _hex(fam[0].get_fill_color()) if fam else _hex(m.get_color())
        spec = {
            "c": "text",
            "s": m.original_text if hasattr(m, "original_text") else m.text,
            "color": color,
            "bold": getattr(m, "weight", "NORMAL") == "BOLD",
        }
    elif isinstance(m, DashedVMobject):
        spec = {
            "c": "dashrect",
            "stroke": _hex(m.get_stroke_color()),
            "sw": float(m.stroke_width or 1.0),
        }
    elif isinstance(m, (Square, Dot)):
        spec = {
            "c": "token" if isinstance(m, Square) else "dot",
            "fill": _hex(m.get_fill_color()),
            "fo": float(m.fill_opacity or 0),
            "stroke": _hex(m.get_stroke_color()),
            "sw": float(m.stroke_width or 0),
        }
    elif isinstance(m, Rectangle):  # incl. RoundedRectangle, SurroundingRectangle
        spec = {
            "c": "rect",
            "fill": _hex(m.get_fill_color()),
            "fo": round(float(m.fill_opacity or 0), 3),
            "stroke": _hex(m.get_stroke_color()),
            "so": round(float(m.stroke_opacity or 0), 3),
            "sw": float(m.stroke_width or 0),
            "r": round(float(getattr(m, "corner_radius", 0) or 0), 3),
        }
    elif isinstance(m, Line):  # incl. DashedLine (its dashes are submobjects)
        s0, s1 = m.get_start(), m.get_end()
        spec = {
            "c": "line",
            "dx": round(float(s1[0] - s0[0]), 3),
            "dy": round(float(s1[1] - s0[1]), 3),
            "stroke": _hex(m.get_stroke_color()),
            "sw": float(m.stroke_width or 1),
            "dashed": type(m).__name__ == "DashedLine",
        }
    elif isinstance(m, Arc):
        spec = {
            "c": "arc",
            "stroke": _hex(m.get_stroke_color()),
            "sw": float(m.stroke_width or 1),
        }
    if spec is None:
        UNSERIALIZED.append(type(m).__name__)
        return None
    st = state(m)
    spec["w0"] = st["w"] or 0.0001
    spec["h0"] = round(float(m.height), 4) if m.height else 0.0001
    spec["o0"] = st["o"]
    spec["x0"], spec["y0"] = st["x"], st["y"]
    spec["z"] = int(getattr(m, "z_index", 0) or 0)
    if getattr(m, "web_save", False):
        spec["sv"] = 1
    return spec
