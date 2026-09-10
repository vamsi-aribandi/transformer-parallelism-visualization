"""Device boxes and the mesh grid. Relative layout only — no absolute coords."""

from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Rectangle, RoundedRectangle, Text, VGroup

from tpviz import style
from tpviz.core.mesh import Mesh


class DeviceBox(VGroup):
    """Rounded rect with a title and two stacked slot regions.

    Slots: "weights" (upper region) and "acts" (lower region). `fit_into` scales
    a mobject group to fit a slot and arranges it there.
    """

    def __init__(self, index: int, *, title: str | None = None,
                 width: float = 3.0, height: float = 4.8):
        super().__init__()
        self.index = index
        hue = style.DEVICE_HUES[index % len(style.DEVICE_HUES)]
        self.hue = hue

        box = RoundedRectangle(corner_radius=0.12, width=width, height=height, stroke_width=2.2)
        box.set_fill(style.DEVICE_BOX_FILL, opacity=1.0)
        box.set_stroke(hue, opacity=0.9)
        box.set_z_index(style.Z_DEVICE)
        self.box = box

        label = Text(title or f"Device {index}", font_size=20, color=hue, weight="BOLD")
        label.set_z_index(style.Z_LABEL)
        label.next_to(box.get_top(), DOWN, buff=0.14)
        self.title = label
        self.add(box, label)

    def _slot_frame(self, slot: str) -> tuple[np.ndarray, float, float]:
        """(center, width, height) of a slot, computed from the box's CURRENT
        position — slots must survive the grid arranging/shifting the boxes."""
        top = self.box.get_top()[1] - 0.55  # leave room for the title
        bottom = self.box.get_bottom()[1] + 0.15
        mid = (top + bottom) / 2
        cx = self.box.get_center()[0]
        inner_w = self.box.width - 0.35
        if slot == "weights":
            center = np.array([cx, (top + mid) / 2, 0.0])
            return center, inner_w, top - mid - 0.1
        if slot == "acts":
            center = np.array([cx, (mid + bottom) / 2, 0.0])
            return center, inner_w, mid - bottom - 0.1
        raise KeyError(slot)

    def slot_center(self, slot: str) -> np.ndarray:
        return self._slot_frame(slot)[0]

    def fit_into(self, group: VGroup, slot: str, *, max_scale: float = 1.0,
                 attach: bool = True) -> VGroup:
        """Scale `group` down (never up beyond max_scale) to fit the slot, center it.

        attach=True parents the group to this DeviceBox (static layouts); scenes
        that animate tensors independently pass attach=False and add to the
        scene themselves.
        """
        center, fw, fh = self._slot_frame(slot)
        s = min(
            max_scale,
            fw / max(group.width, 1e-6),
            fh / max(group.height, 1e-6),
        )
        group.scale(s)
        group.move_to(center)
        if attach:
            self.add(group)
        return group


class DeviceGrid(VGroup):
    """Row of DeviceBoxes for a 1D mesh (grid layouts arrive with 2D meshes)."""

    def __init__(self, mesh: Mesh, *, box_width: float = 3.0, box_height: float = 4.8,
                 buff: float = 0.35, titles: list[str] | None = None):
        super().__init__()
        self.mesh = mesh
        n = mesh.n_devices
        self.boxes = [
            DeviceBox(
                i,
                title=titles[i] if titles else None,
                width=box_width,
                height=box_height,
            )
            for i in range(n)
        ]
        row = VGroup(*self.boxes)
        row.arrange(RIGHT, buff=buff)
        self.add(row)

    def box(self, i: int) -> DeviceBox:
        return self.boxes[i]

    def __len__(self) -> int:
        return len(self.boxes)
