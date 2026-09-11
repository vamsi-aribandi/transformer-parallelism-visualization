"""Visual tensors.

- ActivationDeck: [B, T, D]-shaped tensors as a fake-3D deck of B slabs, each
  slab a T x D rectangle. Sharding renders as: B -> this device's slab solid
  within a ghost stack, T -> horizontal band of each slab, D -> vertical slice.
  The full extent stays visible as a dashed ghost outline, and the solid part
  sits at THIS device's offset within it, so different devices visibly hold
  different slices.
- WeightRect: 2D weights as flat hatched rectangles; aspect follows the dims
  ([D, F] is wide, [F, D] is tall). Sharded dims use the same ghost treatment.

Everything is plain 2D (no ThreeDScene) — the deck is an offset illusion.
"""

from __future__ import annotations

import numpy as np
from manim import (
    DashedVMobject,
    Line,
    Rectangle,
    VGroup,
)

from tpviz import style
from tpviz.core.mesh import Mesh
from tpviz.core.tensors import LTensor
from tpviz.mobjects.labels import tensor_label

# Visual length (in scene units) of each logical dim at scale 1.
DIM_LEN = {"D": 1.5, "F": 2.3, "H": 1.2, "T": 1.0, "S": 0.9, "B": 1.0, "E": 1.0}
N_SLABS = 4  # visual slab count for an unsharded B
SLAB_DX, SLAB_DY = 0.09, 0.07

TOP_LEFT = np.array([-1.0, 1.0, 0.0])


def axis_coord(mesh: Mesh, axis: str, device: int) -> int:
    """Device's coordinate along `axis`, unraveling the flat index row-major
    over the mesh axes (1D meshes: coordinate == device index)."""
    rem = device
    for name, size in reversed(list(mesh.axes.items())):
        coord = rem % size
        if name == axis:
            return coord
        rem //= size
    raise KeyError(axis)


def _shard_fraction(t: LTensor, mesh: Mesh, dim: str, device: int) -> tuple[float, float]:
    """(fraction of the dim this device holds, offset fraction of its shard)."""
    axis = t.axis_of(dim)
    if axis is None:
        return 1.0, 0.0
    n = mesh.size(axis)
    return 1.0 / n, axis_coord(mesh, axis, device) / n


def _ghost(width: float, height: float, *, num_dashes: int = 36) -> DashedVMobject:
    rect = Rectangle(width=width, height=height, stroke_width=1.4)
    rect.set_stroke(style.MUTED_TEXT, opacity=0.45)
    rect.set_fill(opacity=0.0)
    return DashedVMobject(rect, num_dashes=num_dashes)


def _hatch(rect: Rectangle, spacing: float = 0.18) -> VGroup:
    """Thin diagonal lines clipped to the rect — the weight texture."""
    lines = VGroup()
    w, h = rect.width, rect.height
    x0, y0 = rect.get_left()[0], rect.get_bottom()[1]
    # 45-degree lines: x + y = c for c in range
    c = x0 + y0 + spacing
    while c < x0 + w + y0 + h:
        xa, ya = max(x0, c - (y0 + h)), min(y0 + h, c - x0)
        xb, yb = min(x0 + w, c - y0), max(y0, c - (x0 + w))
        if xa < xb:
            line = Line([xa, ya, 0], [xb, yb, 0], stroke_width=0.8)
            line.set_stroke(style.WEIGHT_STROKE, opacity=0.25)
            lines.add(line)
        c += spacing
    return lines


def _place_in_ghost(solid, ghost, full_w: float, full_h: float,
                    fw: float, ow: float, fh: float, oh: float) -> None:
    """Anchor `solid` inside `ghost` at its shard offsets (ow rightward from
    the left edge, oh downward from the top edge, as fractions)."""
    solid.move_to(ghost.get_corner(TOP_LEFT), aligned_edge=TOP_LEFT)
    solid.shift(np.array([full_w * ow, -full_h * oh, 0.0]))


class TensorVis(VGroup):
    """Base: holds the logical tensor and offers a notation label."""

    def __init__(self, t: LTensor, mesh: Mesh, *, device: int = 0,
                 sharded: bool = True, scale: float = 1.0):
        super().__init__()
        self.t = t
        self.mesh = mesh
        self.device = device
        self.sharded = sharded  # False -> draw the full logical tensor
        self.vis_scale = scale

    def frac(self, dim: str) -> tuple[float, float]:
        if not self.sharded:
            return 1.0, 0.0
        return _shard_fraction(self.t, self.mesh, dim, self.device)

    def make_label(self, font_size: float = 22) -> VGroup:
        label = tensor_label(self.t, font_size=font_size)
        label.next_to(self, direction=np.array([0.0, -1.0, 0.0]), buff=0.12)
        return label


class WeightRect(TensorVis):
    def __init__(self, t: LTensor, mesh: Mesh, *, device: int = 0,
                 sharded: bool = True, scale: float = 1.0):
        super().__init__(t, mesh, device=device, sharded=sharded, scale=scale)
        d0, d1 = t.dims[-2], t.dims[-1]  # MoE weights [E, D, F] draw their [D, F] face
        full_h = DIM_LEN[d0] * 0.75 * scale
        full_w = DIM_LEN[d1] * 0.75 * scale

        fill = style.TENSOR_FILL.get(t.kind, style.WEIGHT_FILL)
        stroke = style.TENSOR_STROKE.get(t.kind, style.WEIGHT_STROKE)
        f0, o0 = self.frac(d0)
        f1, o1 = self.frac(d1)
        solid = Rectangle(width=full_w * f1, height=full_h * f0, stroke_width=1.8)
        opacity = style.PARTIAL_FILL_OPACITY if t.partial else style.FILL_OPACITY
        solid.set_fill(fill, opacity=opacity)
        solid.set_stroke(stroke)
        self.solid = solid

        if f0 < 1.0 or f1 < 1.0:
            ghost = _ghost(full_w, full_h)
            _place_in_ghost(solid, ghost, full_w, full_h, f1, o1, f0, o0)
            self.add(ghost, solid, _hatch(solid))
        else:
            self.add(solid, _hatch(solid))
        self.set_z_index(style.Z_TENSOR)


class ActivationDeck(TensorVis):
    def __init__(self, t: LTensor, mesh: Mesh, *, device: int = 0,
                 sharded: bool = True, scale: float = 1.0):
        super().__init__(t, mesh, device=device, sharded=sharded, scale=scale)
        width_dim = t.dims[-1]  # D / F / H
        full_w = DIM_LEN[width_dim] * scale
        full_h = DIM_LEN["T"] * scale

        fb, ob = self.frac("B") if "B" in t.dims else (1.0, 0.0)
        ft, ot = self.frac("T") if "T" in t.dims else (1.0, 0.0)
        fw, ow = self.frac(width_dim)

        fill = style.TENSOR_FILL[t.kind]
        stroke = style.TENSOR_STROKE[t.kind]
        opacity = style.PARTIAL_FILL_OPACITY if t.partial else style.FILL_OPACITY

        # Which stack positions hold THIS device's slabs. Unsharded B: all of
        # them. Sharded B: a contiguous run at the device's offset; the other
        # positions render as dashed ghost slabs so devices visibly differ.
        if "B" in t.dims:
            n_own = max(1, round(N_SLABS * fb))
            own_start = min(round(N_SLABS * ob), N_SLABS - n_own)
            own = set(range(own_start, own_start + n_own))
        else:
            own = {0}
        total = N_SLABS if ("B" in t.dims and fb < 1.0) else max(own) + 1

        slabs = VGroup()
        front_own = min(own)
        for i in range(total - 1, -1, -1):  # back to front
            offset = np.array([SLAB_DX, SLAB_DY, 0.0]) * i * scale
            slab = VGroup()
            if i not in own:
                ghost_slab = _ghost(full_w, full_h, num_dashes=24)
                slab.add(ghost_slab)
                slab.shift(offset)
                slabs.add(slab)
                continue
            depth_fade = 0.55 if i != front_own else 1.0
            solid = Rectangle(width=full_w * fw, height=full_h * ft, stroke_width=1.6)
            solid.set_fill(fill, opacity=opacity * depth_fade)
            solid.set_stroke(stroke, opacity=depth_fade)
            if t.partial:
                # dashed outline on top of a translucent fill = "not yet a real number"
                dashes = DashedVMobject(
                    Rectangle(width=solid.width, height=solid.height, stroke_width=1.6),
                    num_dashes=28,
                )
                dashes.set_stroke(stroke, opacity=depth_fade)
                dashes.set_fill(opacity=0.0)
                solid.set_stroke(opacity=0.0)
                solid = VGroup(solid, dashes)
            # ghost outline of the full logical extent on the front owned slab only
            if (ft < 1.0 or fw < 1.0) and i == front_own:
                ghost = _ghost(full_w, full_h)
                _place_in_ghost(solid, ghost, full_w, full_h, fw, ow, ft, ot)
                slab.add(ghost, solid)
            else:
                if ft < 1.0 or fw < 1.0:
                    # align back slabs with the front slab's shard position
                    solid.shift(
                        np.array(
                            [
                                -(full_w - full_w * fw) / 2 + full_w * ow,
                                (full_h - full_h * ft) / 2 - full_h * ot,
                                0.0,
                            ]
                        )
                    )
                slab.add(solid)
            slab.shift(offset)
            slabs.add(slab)
        self.slabs = slabs
        self.add(slabs)
        self.set_z_index(style.Z_TENSOR)
