"""Pipeline parallelism on the shared model canvas.

Stage lanes own their layer's stations (non-owned stations are dimmed); the
microbatch queue waits in the In dock, hops diagonally down-right at the stage
boundary (the P2P send), and retires into the Out dock. Ticks are played as
grouped animations so the overlap (stage 0 on mb1 while stage 1 runs mb0) is
genuinely simultaneous. A Gantt inset below the lanes builds from the same
tick fields the scheduler emitted.
"""

from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    DashedVMobject,
    FadeIn,
    Indicate,
    LaggedStart,
    Rectangle,
    SurroundingRectangle,
    Text,
    VGroup,
)

from tpviz import configs, style
from tpviz.core.model import forward_steps, make_input, make_weight
from tpviz.core.steps import P2PSendStep, StageComputeStep
from tpviz.mobjects.canvas import DECK_SCALE, FIXTURE_SCALE, ModelCanvas
from tpviz.mobjects.equation_strip import EquationStrip
from tpviz.mobjects.labels import caption_text
from tpviz.mobjects.tensor_mobject import ActivationDeck, WeightRect
from tpviz.scenes.base import PHASE_WEIGHTS, ForwardPassScene

CELL_W, CELL_H = 0.62, 0.34  # defaults; scenes may override via class attrs


class PPScene(ForwardPassScene):
    cfg = configs.PP
    cell_w = CELL_W
    cell_h = CELL_H

    def construct(self):
        self.speed = 1.0
        self.comm_count = 0
        self.acts, self.weights, self.pending_restore = {}, {}, {}
        self.caption_mobj, self.badge, self.tracker, self.highlight = None, None, None, None

        self.next_section("title")
        self.title_card()
        self.next_section("setup")
        self.setup_stage()
        self.comm_label.become(
            caption_text("P2P sends", font_size=16, color=style.MUTED_TEXT).move_to(
                self.comm_label.get_center()
            )
        )
        self.place_initial_tensors()
        self.next_section("pipeline")
        self.build_gantt()
        self.run_pipeline()
        self.next_section("bubble")
        self.bubble_beat()
        self.next_section("summary")
        self.summary()

    def make_canvas(self) -> ModelCanvas:
        # shorter lanes: the band below is for the Gantt inset
        return ModelCanvas(self.cfg, bottom=-1.75, lane_buff=0.15)

    # ------------------------------------------------------------------ setup
    def place_initial_tensors(self):
        fixtures, dims = [], []
        for s in self.canvas.stations:
            if s.kind != "block":
                continue
            for lane in range(self.n_lanes()):
                if s.owner is not None and s.owner != lane:
                    dims.append(self.canvas.dim_station(s.key, lane))
                    continue
                for slot, name in enumerate(PHASE_WEIGHTS[s.phase]):
                    t = make_weight(self.cfg, name)
                    vis = WeightRect(t, self.cfg.mesh, device=lane, scale=FIXTURE_SCALE)
                    self.canvas.fit_weight(VGroup(vis), s.key, slot, lane)
                    fixtures.append(vis)
                    self.weights.setdefault((s.key, name), []).append(vis)
        self.show_caption_text(
            "Each stage owns one full layer. The batch splits into microbatches."
        )
        self.play(
            LaggedStart(*[FadeIn(f) for f in fixtures], lag_ratio=0.02),
            *[FadeIn(d) for d in dims],
            run_time=1.2,
        )

        pp = self.cfg.pipeline
        t_in = make_input(self.cfg)
        self.queue: list[VGroup] = []
        for m in range(pp.n_microbatches):
            deck = ActivationDeck(t_in, self.cfg.mesh, scale=DECK_SCALE * 0.85)
            hue = style.MICROBATCH_HUES[m % len(style.MICROBATCH_HUES)]
            frame = SurroundingRectangle(deck, color=hue, stroke_width=2.2, buff=0.05)
            tag = Text(f"mb{m}", font_size=13, color=hue, weight="BOLD")
            tag.next_to(frame, DOWN, buff=0.05)
            g = VGroup(deck, frame, tag)
            if g.height > 0.72:
                g.scale(0.72 / g.height)
            g.move_to(self.canvas.dock_slot("in", m, pitch=0.92))
            g.set_z_index(style.Z_TENSOR)
            self.queue.append(g)
        self.play(
            LaggedStart(*[FadeIn(g, shift=RIGHT * 0.2) for g in self.queue], lag_ratio=0.15),
            run_time=1.1,
        )
        self.stage_deck: dict[int, VGroup | None] = {s: None for s in range(pp.n_stages)}
        self.done: list[VGroup] = []
        self.done_by_mb: dict[int, VGroup] = {}

    # ------------------------------------------------------------------ gantt
    def gantt_n_ticks(self) -> int:
        pp = self.cfg.pipeline
        return pp.n_microbatches + pp.n_stages - 1

    def build_gantt(self):
        pp = self.cfg.pipeline
        n_ticks = self.gantt_n_ticks()
        self.gantt_origin = np.array([0.3 - (n_ticks - 1) * self.cell_w / 2, -2.15, 0.0])

        frame_cells = VGroup()
        for s in range(pp.n_stages):
            for t in range(n_ticks):
                c = Rectangle(width=self.cell_w, height=self.cell_h, stroke_width=1.0)
                c.set_stroke(style.DEVICE_BOX_STROKE, opacity=0.8)
                c.move_to(self.gantt_cell_center(s, t))
                frame_cells.add(c)
        row_labels = VGroup(
            *[
                Text(f"S{s}", font_size=15, color=style.DEVICE_HUES[s]).move_to(
                    self.gantt_cell_center(s, 0) + LEFT * (self.cell_w / 2 + 0.35)
                )
                for s in range(pp.n_stages)
            ]
        )
        time_label = Text("time →", font_size=14, color=style.MUTED_TEXT)
        time_label.next_to(frame_cells, RIGHT, buff=0.2)
        self.gantt = VGroup(frame_cells, row_labels, time_label)
        self.gantt.set_z_index(style.Z_LABEL)
        self.play(FadeIn(self.gantt), run_time=0.6)

    def gantt_cell_center(self, stage: int, tick: int) -> np.ndarray:
        return self.gantt_origin + np.array([tick * self.cell_w, -stage * (self.cell_h + 0.08), 0.0])

    def gantt_fill(self, stage: int, tick: int, mb: int) -> Rectangle:
        c = Rectangle(width=self.cell_w - 0.06, height=self.cell_h - 0.06, stroke_width=0)
        c.set_fill(style.MICROBATCH_HUES[mb % len(style.MICROBATCH_HUES)], opacity=0.9)
        c.move_to(self.gantt_cell_center(stage, tick))
        c.set_z_index(style.Z_LABEL + 1)
        return c

    # --------------------------------------------------------------- pipeline
    def _attn_key(self, stage: int) -> str:
        return f"l{stage + 1}-attn"

    def _mlp_key(self, stage: int) -> str:
        return f"l{stage + 1}-mlp"

    def run_pipeline(self):
        steps = forward_steps(self.cfg)
        computes = [s for s in steps if isinstance(s, StageComputeStep)]
        sends = [s for s in steps if isinstance(s, P2PSendStep)]
        n_ticks = max(c.tick for c in computes) + 1

        for t in range(n_ticks):
            tick_computes = [c for c in computes if c.tick == t]
            tick_sends = [p for p in sends if p.tick == t]

            # 1) stage 0 loads its next microbatch from the In dock
            for c in tick_computes:
                if c.stage == 0:
                    deck = self.queue.pop(0)
                    self.stage_deck[0] = deck
                    self.play(
                        deck.animate.move_to(
                            self.canvas.anchor(self._attn_key(0), "entry", 0)
                        ),
                        run_time=0.5,
                    )

            if t == 1 and len(tick_computes) == 2:
                self.show_caption_text(
                    "The overlap: stage 0 starts mb1 while stage 1 processes mb0"
                )

            # 2) all active stages traverse their attention then MLP stations —
            #    grouped plays keep the overlap genuinely simultaneous
            for is_attn in (True, False):
                anims, fills = [], []
                for c in tick_computes:
                    deck = self.stage_deck[c.stage]
                    key = self._attn_key(c.stage) if is_attn else self._mlp_key(c.stage)
                    anims.append(
                        deck.animate.move_to(self.canvas.anchor(key, "exit", c.stage))
                    )
                    for name in PHASE_WEIGHTS[self.canvas.station(key).phase]:
                        for vis in self.weights[(key, name)]:
                            anims.append(
                                Indicate(vis, scale_factor=1.05, color=style.WEIGHT_STROKE)
                            )
                    if is_attn:
                        fills.append(FadeIn(self.gantt_fill(c.stage, t, c.microbatch)))
                self.play(*anims, *fills, run_time=0.9)

            # 3) a finished last-stage microbatch retires into the Out dock
            for c in tick_computes:
                if c.stage == self.n_lanes() - 1:
                    deck = self.stage_deck[c.stage]
                    self.stage_deck[c.stage] = None
                    self.done.append(deck)
                    self.done_by_mb[c.microbatch] = deck
                    self.play(
                        deck.animate.move_to(
                            self.canvas.dock_slot("out", len(self.done) - 1, pitch=0.92)
                        ),
                        run_time=0.45,
                    )

            # 4) the P2P handoff: a diagonal down-right hop across the boundary
            for p in tick_sends:
                deck = self.stage_deck[p.src_stage]
                self.stage_deck[p.src_stage] = None
                self.play(self.strip.show(p.tex(), color=style.COMM_COLOR), run_time=0.3)
                self.play(
                    deck.animate(path_arc=-0.35).move_to(
                        self.canvas.anchor(self._attn_key(p.dst_stage), "entry", p.dst_stage)
                    ),
                    run_time=0.65,
                )
                self.stage_deck[p.dst_stage] = deck
                self.comm_bump()

    def bubble_beat(self):
        pp = self.cfg.pipeline
        n_ticks = pp.n_microbatches + pp.n_stages - 1
        empties = []
        for s in range(pp.n_stages):
            for t in range(n_ticks):
                if not (0 <= t - s < pp.n_microbatches):
                    r = Rectangle(width=self.cell_w - 0.06, height=self.cell_h - 0.06, stroke_width=1.8)
                    r.set_stroke(style.COMM_COLOR)
                    r.move_to(self.gantt_cell_center(s, t))
                    r.set_z_index(style.Z_LABEL + 1)
                    empties.append(DashedVMobject(r, num_dashes=16))
        tag = Text(
            "bubble = idle time; more microbatches → smaller fraction",
            font_size=17,
            color=style.COMM_COLOR,
        )
        tag.next_to(self.gantt, UP, buff=0.1)
        tag.set_z_index(style.Z_LABEL + 1)
        self.show_caption_text("Communication: only activation handoffs between adjacent stages")
        self.play(*[FadeIn(e) for e in empties], FadeIn(tag), run_time=0.7)
        self.wait(2.0)

    def summary_row(self, op: str, n: int) -> str:
        return f"{op}  ×{n}   (one per microbatch per stage boundary)"
