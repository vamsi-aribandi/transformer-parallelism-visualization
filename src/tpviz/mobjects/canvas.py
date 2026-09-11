"""Shared model canvas: device lanes (y) x model-depth stations (x).

The input flows LEFT -> RIGHT through stations (In dock, one block station per
layer-phase, Out dock); devices are horizontal lanes stacked vertically, so
collectives fly vertically between lanes at the current station's x. Pipeline
parallelism shares the canvas: each stage lane owns its layer's stations.

Geometry philosophy (matches DeviceBox before it): station spans and track
centers are stored as FRACTIONS and resolved against the invisible frame
rect's CURRENT coordinates — shifting or scaling the canvas never invalidates
an anchor.
"""

from __future__ import annotations

import numpy as np
from manim import (
    DashedLine,
    MathTex,
    Rectangle,
    RoundedRectangle,
    Text,
    Transform,
    VGroup,
)

from tpviz import style
from tpviz.core.mesh import StrategyConfig
from tpviz.mobjects.labels import caption_text, math_label

# scales for things living on the canvas
DECK_SCALE = 0.40
FIXTURE_SCALE = 0.30
MINI_SCALE = 0.28

# act-track anchor offsets, as fractions of a block station's width
ANCHOR_FRACS: dict[str, dict[str, float]] = {
    "attn": {"entry": 0.14, "w1": 0.31, "core": 0.53, "w2": 0.76, "exit": 0.87},
    "mlp": {"entry": 0.14, "w1": 0.34, "mid": 0.53, "w2": 0.72, "exit": 0.87},
    "moe": {"entry": 0.14, "tokens": 0.31, "core": 0.53, "combine": 0.74, "exit": 0.87},
}
# weight-fixture slot x, as fractions of a block station's width (slot 0/1)
WEIGHT_SLOT_FRACS: dict[str, tuple[float, float]] = {
    "attn": (0.31, 0.76),
    "mlp": (0.34, 0.72),
    "moe": (0.47, 0.71),
}
# lane-internal track centers, as fractions of lane height from the TOP
WEIGHT_TRACK = 0.24
ACT_TRACK = 0.70


class StationSpec:
    def __init__(self, key: str, kind: str, layer: int, phase: str, owner: int | None):
        self.key, self.kind, self.layer, self.phase, self.owner = key, kind, layer, phase, owner


class Lane(VGroup):
    """Lane border + `Dev i` tag; track centers computed live from the rect."""

    def __init__(self, index: int, *, width: float, height: float, title: str | None = None):
        super().__init__()
        self.index = index
        self.hue = style.DEVICE_HUES[index % len(style.DEVICE_HUES)]
        self.rect = RoundedRectangle(
            corner_radius=0.08, width=width, height=height, stroke_width=1.8
        )
        self.rect.set_fill(style.DEVICE_BOX_FILL, opacity=1.0)
        self.rect.set_stroke(self.hue, opacity=0.85)
        self.rect.set_z_index(style.Z_DEVICE)
        self.tag = Text(title or f"Dev {index}", font_size=15, color=self.hue, weight="BOLD")
        self.tag.set_z_index(style.Z_LABEL)
        self.add(self.rect, self.tag)

    def place_tag(self, x: float) -> None:
        self.tag.move_to(np.array([x, self.rect.get_center()[1], 0.0]))

    def weight_y(self) -> float:
        return self.rect.get_top()[1] - WEIGHT_TRACK * self.rect.height

    def act_y(self) -> float:
        return self.rect.get_top()[1] - ACT_TRACK * self.rect.height


class ModelCanvas(VGroup):
    def __init__(
        self,
        cfg: StrategyConfig,
        *,
        x_left: float = -7.0,
        x_right: float = 7.0,
        top: float = 1.95,
        bottom: float = -2.6,
        tag_w: float = 0.7,
        dock_w: float = 0.8,
        lane_buff: float = 0.10,
    ):
        super().__init__()
        self.cfg = cfg

        # ---- stations -------------------------------------------------
        mlp_phase = "moe" if cfg.moe is not None else "mlp"
        self.stations: list[StationSpec] = [StationSpec("in", "dock", 0, "", None)]
        for layer in range(1, cfg.n_layers + 1):
            owner = layer - 1 if cfg.pipeline is not None else None
            self.stations.append(StationSpec(f"l{layer}-attn", "block", layer, "attn", owner))
            self.stations.append(
                StationSpec(f"l{layer}-{mlp_phase}", "block", layer, mlp_phase, owner)
            )
        self.stations.append(StationSpec("out", "dock", 0, "", None))
        self._by_key = {s.key: s for s in self.stations}

        n_blocks = 2 * cfg.n_layers
        usable = (x_right - x_left) - tag_w - 2 * dock_w
        block_w = usable / n_blocks

        # frame rect: the live geometry reference (invisible)
        self.frame_rect = Rectangle(
            width=x_right - x_left, height=top - bottom, stroke_width=0
        )
        self.frame_rect.set_fill(opacity=0.0)
        self.frame_rect.move_to(np.array([(x_left + x_right) / 2, (top + bottom) / 2, 0.0]))
        self.add(self.frame_rect)

        # station x-fractions of the frame width
        self._fracs: dict[str, tuple[float, float]] = {}
        fw = x_right - x_left
        x = tag_w
        for s in self.stations:
            w = dock_w if s.kind == "dock" else block_w
            self._fracs[s.key] = (x / fw, (x + w) / fw)
            x += w

        # ---- lanes -----------------------------------------------------
        if cfg.pipeline is not None:
            n_lanes = cfg.pipeline.n_stages
            titles = [f"Stage {s}" for s in range(n_lanes)]
            # PP lanes span the block stations only; docks are canvas columns
            lane_x0 = x_left + tag_w + dock_w
            lane_x1 = x_right - dock_w
        else:
            n_lanes = cfg.mesh.n_devices
            titles = None
            lane_x0 = x_left + tag_w
            lane_x1 = x_right
        lane_h = ((top - bottom) - (n_lanes - 1) * lane_buff) / n_lanes
        self.lanes: list[Lane] = []
        for i in range(n_lanes):
            lane = Lane(
                i,
                width=lane_x1 - lane_x0,
                height=lane_h,
                title=titles[i] if titles else None,
            )
            lane_top = top - i * (lane_h + lane_buff)
            lane.rect.move_to(
                np.array([(lane_x0 + lane_x1) / 2, lane_top - lane_h / 2, 0.0])
            )
            lane.place_tag(x_left + tag_w / 2)
            self.lanes.append(lane)
            self.add(lane)

        # ---- station separators (dashed verticals at block boundaries) --
        seps = VGroup()
        for s in self.stations[1:-1]:
            x0, _ = self.station_span(s.key)
            line = DashedLine(
                np.array([x0, top, 0.0]), np.array([x0, bottom, 0.0]), stroke_width=1.0
            )
            line.set_stroke(style.DEVICE_BOX_STROKE, opacity=0.7)
            seps.add(line)
        x1_last, _ = self._fracs["out"]
        xb = x_left + x1_last * fw
        line = DashedLine(np.array([xb, top, 0.0]), np.array([xb, bottom, 0.0]), stroke_width=1.0)
        line.set_stroke(style.DEVICE_BOX_STROKE, opacity=0.7)
        seps.add(line)
        seps.set_z_index(style.Z_DEVICE + 1)
        self.separators = seps
        self.add(seps)

        # ---- header band -------------------------------------------------
        self.header_titles = VGroup()
        self._header_tex: dict[str, MathTex] = {}
        self.header_texs = VGroup()
        for s in self.stations:
            cx = sum(self.station_span(s.key)) / 2
            if s.kind == "dock":
                name = {"in": "In", "out": "Out"}[s.key]
            else:
                name = f"Layer {s.layer} · {self._phase_title(s.phase)}"
            t = caption_text(name, font_size=17, color=style.TEXT_COLOR)
            if s.kind == "dock":
                t.set_color(style.MUTED_TEXT)
            t.move_to(np.array([cx, top + 0.72, 0.0]))
            t.set_z_index(style.Z_LABEL)
            self.header_titles.add(t)
        self.add(self.header_titles)

    @staticmethod
    def _phase_title(phase: str) -> str:
        return {"attn": "Attention", "mlp": "MLP", "moe": "MoE"}[phase]

    # ------------------------------------------------------------ geometry
    def station(self, key: str) -> StationSpec:
        return self._by_key[key]

    def station_key(self, layer: int, phase: str) -> str:
        return f"l{layer}-{phase}"

    def station_span(self, key: str) -> tuple[float, float]:
        f0, f1 = self._fracs[key]
        left = self.frame_rect.get_left()[0]
        w = self.frame_rect.width
        return left + f0 * w, left + f1 * w

    def anchor(self, key: str, name: str, lane: int) -> np.ndarray:
        x0, x1 = self.station_span(key)
        s = self._by_key[key]
        if s.kind == "dock":
            x = (x0 + x1) / 2
        else:
            x = x0 + ANCHOR_FRACS[s.phase][name] * (x1 - x0)
        return np.array([x, self.lanes[lane].act_y(), 0.0])

    def weight_anchor(self, key: str, slot: int, lane: int) -> np.ndarray:
        x0, x1 = self.station_span(key)
        s = self._by_key[key]
        x = x0 + WEIGHT_SLOT_FRACS[s.phase][slot] * (x1 - x0)
        return np.array([x, self.lanes[lane].weight_y(), 0.0])

    def stash_slot(self, key: str, lane: int, anchor: str, spread: int = 0) -> np.ndarray:
        """Where a saved activation parks: on the lane's bottom edge, directly
        under the weight whose backward (dW) matmul will consume it.
        anchor: "slot0" | "slot1" (weight fixtures) or "core" (Q/K/V, spread apart).
        """
        if anchor in ("slot0", "slot1"):
            x = self.weight_anchor(key, 0 if anchor == "slot0" else 1, lane)[0]
        else:
            x = self.anchor(key, anchor, lane)[0]
        x += spread * 0.36
        rect = self.lanes[lane].rect
        return np.array([x, rect.get_bottom()[1] + 0.15, 0.0])

    def grad_anchor(self, key: str, slot: int, lane: int) -> np.ndarray:
        """Where a weight-gradient rect sits: a 'shadow' offset from its weight."""
        return self.weight_anchor(key, slot, lane) + np.array([0.16, -0.15, 0.0])

    def dock_slot(self, side: str, i: int, *, pitch: float = 0.80) -> np.ndarray:
        x0, x1 = self.station_span(side)
        top_y = self.frame_rect.get_top()[1]
        return np.array([(x0 + x1) / 2, top_y - 0.45 - i * pitch, 0.0])

    # ---------------------------------------------------------- placement
    def fit_act(self, group: VGroup, key: str, name: str, lane: int,
                *, max_w: float = 0.95, max_h: float | None = None) -> VGroup:
        lane_h = self.lanes[lane].rect.height
        cap_h = max_h if max_h is not None else lane_h * 0.52
        s = min(1.0, max_w / max(group.width, 1e-6), cap_h / max(group.height, 1e-6))
        group.scale(s)
        group.move_to(self.anchor(key, name, lane))
        return group

    def fit_weight(self, group: VGroup, key: str, slot: int, lane: int,
                   *, max_w: float = 0.62, max_h: float | None = None) -> VGroup:
        lane_h = self.lanes[lane].rect.height
        cap_h = max_h if max_h is not None else lane_h * 0.40
        s = min(1.0, max_w / max(group.width, 1e-6), cap_h / max(group.height, 1e-6))
        group.scale(s)
        group.move_to(self.weight_anchor(key, slot, lane))
        return group

    # -------------------------------------------------------------- header
    def add_header_tex(self, key: str, tex: str) -> MathTex:
        """The once-per-station weight notation line under the station title."""
        x0, x1 = self.station_span(key)
        m = math_label(tex, font_size=20, color=style.MUTED_TEXT)
        if m.width > (x1 - x0) - 0.2:
            m.scale(((x1 - x0) - 0.2) / m.width)
        top_y = self.frame_rect.get_top()[1]
        m.move_to(np.array([(x0 + x1) / 2, top_y + 0.33, 0.0]))
        m.set_z_index(style.Z_LABEL)
        self._header_tex[key] = m
        self.header_texs.add(m)
        self.add(m)
        return m

    def set_header_weight_tex(self, key: str, tex: str) -> Transform:
        """Swap a station's header notation (FSDP jit gather/restore)."""
        old = self._header_tex[key]
        x0, x1 = self.station_span(key)
        new = math_label(tex, font_size=20, color=style.MUTED_TEXT)
        if new.width > (x1 - x0) - 0.2:
            new.scale(((x1 - x0) - 0.2) / new.width)
        new.move_to(old.get_center())
        new.set_z_index(style.Z_LABEL)
        self._header_tex[key] = new
        self.header_texs.remove(old)
        self.header_texs.add(new)
        self.remove(old)
        self.add(new)
        return Transform(old, new, replace_mobject_with_target_in_scene=True)

    # ------------------------------------------------------------- overlays
    def station_highlight(self, key: str) -> Rectangle:
        x0, x1 = self.station_span(key)
        top_y = self.frame_rect.get_top()[1]
        bot_y = self.frame_rect.get_bottom()[1]
        r = Rectangle(width=x1 - x0, height=top_y - bot_y, stroke_width=0)
        r.set_fill(style.ACCENT, opacity=0.06)
        r.move_to(np.array([(x0 + x1) / 2, (top_y + bot_y) / 2, 0.0]))
        r.set_z_index(style.Z_DEVICE + 2)
        return r

    def dim_station(self, key: str, lane: int) -> Rectangle:
        x0, x1 = self.station_span(key)
        rect = self.lanes[lane].rect
        r = Rectangle(width=x1 - x0 - 0.06, height=rect.height - 0.08, stroke_width=0)
        r.set_fill("#000000", opacity=0.45)
        r.move_to(np.array([(x0 + x1) / 2, rect.get_center()[1], 0.0]))
        r.set_z_index(style.Z_DEVICE + 3)
        return r
