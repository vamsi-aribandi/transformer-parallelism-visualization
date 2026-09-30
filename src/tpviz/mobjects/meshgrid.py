"""MeshGrid: the multi-axis device canvas for the 5D combination.

Thirty-two devices don't fit as lanes, so a composed mesh is drawn as one
device GRID per pipeline stage, side by side (model depth still runs left ->
right: stage 0 = layer 1, stage 1 = layer 2). Inside a stage the 16 devices of
X x Y x C x Z tile a nested 4 x 4:

    rows = (X outer, C inner)     cols = (Z outer, Y inner)

so every mesh axis is a fixed geometric relation between boxes — FSDP (X)
partners are two rows apart, context (C) partners are adjacent rows, expert
(Z) partners two columns apart, tensor (Y) partners adjacent columns, and
pipeline partners are the same cell in the other grid. A collective over an
axis is therefore recognizable from the flight geometry alone, before the
axis color or the badge says so.

Each box is a miniature lane: weight fixtures on an upper track, the
traveling activation deck on a lower track. Geometry is resolved live from the
box rects (never cached at construction).
"""

from __future__ import annotations

import math

import numpy as np
from manim import Rectangle, RoundedRectangle, Text, VGroup

from tpviz import style
from tpviz.core.mesh import Mesh, StrategyConfig
from tpviz.mobjects.labels import caption_text

# box-internal anchors as fractions of the box width (a tiny lane: the
# activation drifts right through the forward, left through the backward)
BOX_ANCHORS = {"entry": 0.28, "w1": 0.36, "core": 0.50, "w2": 0.64, "exit": 0.72}
WEIGHT_SLOTS = (0.36, 0.64)
WEIGHT_TRACK = 0.25  # fraction of box height from the top
ACT_TRACK = 0.69

ROW_AXES = ("X", "C")  # outer, inner — whichever of them the mesh has
COL_AXES = ("Z", "Y")


def _cell_offsets(sizes: list[int], cell: float, inner_gap: float, outer_gap: float) -> list[float]:
    """Start offset of each cell along one direction for nested axes (outer
    first): an outer-coordinate change opens the wide gap, otherwise the
    narrow one. A single axis uses the wide gap."""
    n = 1
    for k in sizes:
        n *= k
    out, pos = [], 0.0
    inner = sizes[-1] if len(sizes) > 1 else 1
    for i in range(n):
        out.append(pos)
        if i + 1 < n:
            pos += cell + (inner_gap if (i + 1) % inner else outer_gap)
    return out


class MeshGrid(VGroup):
    def __init__(
        self,
        cfg: StrategyConfig,
        *,
        x_left: float = -7.0,
        x_right: float = 7.0,
        top: float = 1.72,
        bottom: float = -2.52,
        margin_w: float = 0.62,
        stage_gap: float = 0.62,
        inner_gap: float = 0.07,
        outer_gap: float = 0.24,
    ):
        super().__init__()
        self.cfg = cfg
        self.mesh: Mesh = cfg.mesh
        m = self.mesh.axes
        self.n_stages = m.get("stage", 1)
        self.top, self.bottom = top, bottom
        self.row_axes = [a for a in ROW_AXES if a in m]
        self.col_axes = [a for a in COL_AXES if a in m]
        row_sizes = [m[a] for a in self.row_axes] or [1]
        col_sizes = [m[a] for a in self.col_axes] or [1]
        n_rows = math.prod(row_sizes)
        n_cols = math.prod(col_sizes)

        def gaps_total(sizes: list[int]) -> float:
            n = math.prod(sizes)
            inner = sizes[-1] if len(sizes) > 1 else 1
            return sum(inner_gap if (i + 1) % inner else outer_gap for i in range(n - 1))

        grid_w = ((x_right - x_left) - 2 * margin_w - (self.n_stages - 1) * stage_gap) / self.n_stages
        box_w = (grid_w - gaps_total(col_sizes)) / n_cols
        box_h = ((top - bottom) - gaps_total(row_sizes)) / n_rows
        self.box_w, self.box_h = box_w, box_h
        self.grid_w = grid_w
        col_off = _cell_offsets(col_sizes, box_w, inner_gap, outer_gap)
        row_off = _cell_offsets(row_sizes, box_h, inner_gap, outer_gap)

        def cell_index(coords: dict[str, int], axes: list[str]) -> int:
            idx = 0
            for a in axes:
                idx = idx * m[a] + coords[a]
            return idx

        def cell_coords(idx: int, axes: list[str]) -> list[tuple[str, int]]:
            out = []
            for a in reversed(axes):
                out.append((a, idx % m[a]))
                idx //= m[a]
            return list(reversed(out))

        def label_row(items, cx, cy):
            n = len(items)
            for k, (axis, val) in enumerate(items):
                t = Text(f"{axis}{val}", font_size=11, color=style.AXIS_HUES[axis], weight="BOLD")
                t.move_to(np.array([cx + (k - (n - 1) / 2) * 0.36, cy, 0.0]))
                t.set_z_index(style.Z_LABEL)
                self.margin_labels[axis].append(t)
                self.add(t)

        self.grid_x0: list[float] = []
        self.boxes: dict[int, RoundedRectangle] = {}
        self.stage_frames: list[Rectangle] = []
        self.margin_labels: dict[str, list[Text]] = {a: [] for a in ROW_AXES + COL_AXES}
        for s in range(self.n_stages):
            gx0 = x_left + margin_w + s * (grid_w + stage_gap)
            self.grid_x0.append(gx0)
            frame = Rectangle(width=grid_w + 0.16, height=(top - bottom) + 0.16, stroke_width=0)
            frame.set_fill(style.ACCENT, opacity=0.0)
            frame.move_to(np.array([gx0 + grid_w / 2, (top + bottom) / 2, 0.0]))
            frame.set_z_index(style.Z_DEVICE - 1)
            self.stage_frames.append(frame)
            self.add(frame)
            for dev in range(self.mesh.n_devices):
                c = self.mesh.coords(dev)
                if c.get("stage", 0) != s:
                    continue
                r = cell_index(c, self.row_axes)
                k = cell_index(c, self.col_axes)
                box = RoundedRectangle(corner_radius=0.06, width=box_w, height=box_h,
                                       stroke_width=1.4)
                box.set_fill(style.DEVICE_BOX_FILL, opacity=1.0)
                box.set_stroke(style.DEVICE_BOX_STROKE, opacity=1.0)
                box.move_to(np.array([gx0 + col_off[k] + box_w / 2, top - row_off[r] - box_h / 2, 0.0]))
                box.set_z_index(style.Z_DEVICE)
                self.boxes[dev] = box
                self.add(box)
            # column labels above the grid, axis-colored ("Z0  Y1")
            if self.col_axes:
                for k in range(n_cols):
                    label_row(cell_coords(k, self.col_axes), gx0 + col_off[k] + box_w / 2, top + 0.14)
        # row labels in the outer margins ("X0  C1")
        if self.row_axes:
            for r in range(n_rows):
                cy = top - row_off[r] - box_h / 2
                for mx in (x_left + margin_w / 2, x_right - margin_w / 2):
                    label_row(cell_coords(r, self.row_axes), mx, cy)

        # per-stage header titles (swapped live as the phase changes)
        self.stage_titles: list[Text] = []
        for s in range(self.n_stages):
            t = caption_text(f"Stage {s} · Layer {s + 1}", font_size=18, color=style.TEXT_COLOR)
            t.move_to(np.array([self.stage_center_x(s), top + 0.95, 0.0]))
            self.stage_titles.append(t)
            self.add(t)

    # ------------------------------------------------------------ geometry
    def stage_devs(self, s: int) -> list[int]:
        return [d for d in sorted(self.boxes) if self.mesh.coords(d).get("stage", 0) == s]

    def stage_of(self, dev: int) -> int:
        return self.mesh.coords(dev).get("stage", 0)

    def stage_center_x(self, s: int) -> float:
        return self.grid_x0[s] + self.grid_w / 2

    def group(self, dev: int, axis: str) -> list[int]:
        """Every device sharing all coordinates with `dev` except `axis`."""
        c = self.mesh.coords(dev)
        return [self.mesh.index(**{**c, axis: v}) for v in range(self.mesh.size(axis))]

    def peer_on_stage(self, dev: int, s: int) -> int:
        return self.mesh.index(**{**self.mesh.coords(dev), "stage": s})

    def box(self, dev: int) -> RoundedRectangle:
        return self.boxes[dev]

    def anchor(self, dev: int, name: str) -> np.ndarray:
        b = self.boxes[dev]
        x = b.get_left()[0] + BOX_ANCHORS[name] * b.width
        return np.array([x, b.get_top()[1] - ACT_TRACK * b.height, 0.0])

    def weight_anchor(self, dev: int, slot: int) -> np.ndarray:
        b = self.boxes[dev]
        x = b.get_left()[0] + WEIGHT_SLOTS[slot] * b.width
        return np.array([x, b.get_top()[1] - WEIGHT_TRACK * b.height, 0.0])

    def grad_anchor(self, dev: int, slot: int) -> np.ndarray:
        return self.weight_anchor(dev, slot) + np.array([0.14, -0.12, 0.0])

    def header_y(self) -> float:
        return self.top + 0.55

    def badge_pos(self, s: int) -> np.ndarray:
        return np.array([self.stage_center_x(s), self.bottom - 0.26, 0.0])

    # ---------------------------------------------------------- placement
    def fit_act(self, group: VGroup, dev: int, name: str, *, max_w: float | None = None,
                max_h: float | None = None) -> VGroup:
        cap_w = max_w if max_w is not None else self.box_w * 0.5
        cap_h = max_h if max_h is not None else self.box_h * 0.47
        s = min(1.0, cap_w / max(group.width, 1e-6), cap_h / max(group.height, 1e-6))
        group.scale(s)
        group.move_to(self.anchor(dev, name))
        return group

    def fit_weight(self, group: VGroup, dev: int, slot: int) -> VGroup:
        cap_w, cap_h = self.box_w * 0.4, self.box_h * 0.38
        s = min(1.0, cap_w / max(group.width, 1e-6), cap_h / max(group.height, 1e-6))
        group.scale(s)
        group.move_to(self.weight_anchor(dev, slot))
        return group

    # ----------------------------------------------------------- overlays
    def stage_highlight(self, s: int) -> Rectangle:
        f = self.stage_frames[s]
        r = Rectangle(width=f.width, height=f.height, stroke_width=0)
        r.set_fill(style.ACCENT, opacity=0.07)
        r.move_to(f.get_center())
        r.set_z_index(style.Z_DEVICE - 1)
        return r
