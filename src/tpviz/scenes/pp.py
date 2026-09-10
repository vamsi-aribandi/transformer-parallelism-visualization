"""Pipeline parallelism: stages own whole layers; microbatches keep both busy.

This scene bypasses the generic step player: pipeline steps are grouped by
tick and played TOGETHER, so the overlap (stage 0 on mb1 while stage 1 runs
mb0) is genuinely simultaneous on screen. A Gantt inset builds up from the
same tick fields the scheduler emitted, so schedule and animation can't drift.
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
    FadeOut,
    Indicate,
    LaggedStart,
    Rectangle,
    SurroundingRectangle,
    Text,
    VGroup,
)

from tpviz import configs, style
from tpviz.core.model import forward_steps, make_input
from tpviz.core.steps import P2PSendStep, StageComputeStep
from tpviz.mobjects.device_grid import DeviceGrid
from tpviz.mobjects.equation_strip import EquationStrip
from tpviz.mobjects.labels import caption_text
from tpviz.mobjects.tensor_mobject import ActivationDeck
from tpviz.scenes.base import ForwardPassScene

CELL_W, CELL_H = 0.62, 0.34


class PPScene(ForwardPassScene):
    cfg = configs.PP

    def construct(self):
        self.speed = 1.0
        self.comm_count = 0
        self.acts, self.weights, self.pending_restore = {}, {}, {}
        self.caption_mobj, self.badge = None, None

        self.next_section("title")
        self.title_card()
        self.next_section("setup")
        self.setup_stage()
        self.place_initial_tensors()
        self.next_section("pipeline")
        self.build_gantt()
        self.run_pipeline()
        self.next_section("bubble")
        self.bubble_beat()
        self.next_section("summary")
        self.summary()

    # ------------------------------------------------------------------ setup
    def setup_stage(self):
        pp = self.cfg.pipeline
        self.grid = DeviceGrid(
            self.cfg.mesh,
            box_width=4.4,
            box_height=3.1,
            buff=1.6,
            titles=[f"Stage {s}" for s in range(pp.n_stages)],
        )
        self.grid.shift(UP * 0.75)
        self.strip = EquationStrip()
        self.add(self.strip)

        self.comm_label = caption_text("P2P sends", font_size=16, color=style.MUTED_TEXT)
        self.comm_value = caption_text("0", font_size=30, color=style.GOOD_COLOR)
        counter = VGroup(self.comm_label, self.comm_value).arrange(DOWN, buff=0.08)
        counter.to_corner(UP + RIGHT, buff=0.25)
        self.add(counter)
        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in self.grid.boxes], lag_ratio=0.15),
            run_time=1.0,
        )

    def place_initial_tensors(self):
        from tpviz.core.model import make_weight
        from tpviz.mobjects.tensor_mobject import WeightRect

        weight_names = ["W_qkv", "W_o", "W_in", "W_out"]
        tags = []
        for s, box in enumerate(self.grid.boxes):
            cells = []
            for name in weight_names:
                t = make_weight(self.cfg, name)
                vis = WeightRect(t, self.cfg.mesh)
                label = vis.make_label(font_size=30)
                cells.append(VGroup(vis, label))
            grid = VGroup(*cells).arrange_in_grid(rows=1, cols=4, buff=0.3)
            box.fit_into(grid, "weights", attach=False)
            self.weights[f"stage{s}"] = cells
            tag = Text(f"holds all of Layer {s + 1}", font_size=15, color=style.MUTED_TEXT)
            tag.next_to(box.title, DOWN, buff=0.05)
            tag.set_z_index(style.Z_LABEL)
            tags.append(tag)

        self.show_caption_text("Each stage owns one full layer. The batch splits into microbatches.")
        self.play(
            *[FadeIn(c) for s in range(len(self.grid)) for c in self.weights[f"stage{s}"]],
            *[FadeIn(t) for t in tags],
            run_time=1.2,
        )

        # microbatch queue, waiting left of stage 0
        pp = self.cfg.pipeline
        t_in = make_input(self.cfg)
        self.queue: list[VGroup] = []
        for m in range(pp.n_microbatches):
            deck = ActivationDeck(t_in, self.cfg.mesh, scale=0.5)
            hue = style.MICROBATCH_HUES[m % len(style.MICROBATCH_HUES)]
            frame = SurroundingRectangle(deck, color=hue, stroke_width=2.4, buff=0.06)
            tag = Text(f"mb{m}", font_size=15, color=hue, weight="BOLD")
            tag.next_to(frame, DOWN, buff=0.06)
            g = VGroup(deck, frame, tag)
            g.scale(0.75)
            g.set_z_index(style.Z_TENSOR)
            self.queue.append(g)
        col = VGroup(*self.queue).arrange(DOWN, buff=0.22)
        col.next_to(self.grid.box(0), LEFT, buff=0.45)
        self.play(LaggedStart(*[FadeIn(g, shift=RIGHT * 0.2) for g in self.queue], lag_ratio=0.15), run_time=1.2)

        self.stage_deck: dict[int, VGroup | None] = {s: None for s in range(pp.n_stages)}
        self.done: list[VGroup] = []

    # ------------------------------------------------------------------ gantt
    def build_gantt(self):
        pp = self.cfg.pipeline
        n_ticks = pp.n_microbatches + pp.n_stages - 1
        origin = np.array([-n_ticks * CELL_W / 2 + 1.0, -2.05, 0.0])
        self.gantt_origin = origin

        frame_cells = VGroup()
        for s in range(pp.n_stages):
            for t in range(n_ticks):
                c = Rectangle(width=CELL_W, height=CELL_H, stroke_width=1.0)
                c.set_stroke(style.DEVICE_BOX_STROKE, opacity=0.8)
                c.move_to(self.gantt_cell_center(s, t))
                frame_cells.add(c)
        row_labels = VGroup(
            *[
                Text(f"S{s}", font_size=15, color=style.DEVICE_HUES[s]).move_to(
                    self.gantt_cell_center(s, 0) + LEFT * (CELL_W / 2 + 0.35)
                )
                for s in range(pp.n_stages)
            ]
        )
        time_label = Text("time →", font_size=15, color=style.MUTED_TEXT)
        time_label.next_to(frame_cells, DOWN, buff=0.12).align_to(frame_cells, LEFT)
        self.gantt = VGroup(frame_cells, row_labels, time_label)
        self.gantt.set_z_index(style.Z_LABEL)
        self.play(FadeIn(self.gantt), run_time=0.7)

    def gantt_cell_center(self, stage: int, tick: int) -> np.ndarray:
        return self.gantt_origin + np.array([tick * CELL_W, -stage * (CELL_H + 0.1), 0.0])

    def gantt_fill(self, stage: int, tick: int, mb: int) -> VGroup:
        c = Rectangle(width=CELL_W - 0.06, height=CELL_H - 0.06, stroke_width=0)
        c.set_fill(style.MICROBATCH_HUES[mb % len(style.MICROBATCH_HUES)], opacity=0.9)
        c.move_to(self.gantt_cell_center(stage, tick))
        c.set_z_index(style.Z_LABEL + 1)
        return c

    # --------------------------------------------------------------- pipeline
    def run_pipeline(self):
        steps = forward_steps(self.cfg)
        computes = [s for s in steps if isinstance(s, StageComputeStep)]
        sends = [s for s in steps if isinstance(s, P2PSendStep)]
        n_ticks = max(c.tick for c in computes) + 1

        for t in range(n_ticks):
            tick_computes = [c for c in computes if c.tick == t]
            tick_sends = [p for p in sends if p.tick == t]

            # 1) stage 0 loads its next microbatch from the queue
            load_anims = []
            for c in tick_computes:
                if c.stage == 0:
                    deck = self.queue.pop(0)
                    slot = self.grid.box(0).slot_center("acts")
                    load_anims.append(deck.animate.scale(1.25).move_to(slot))
                    self.stage_deck[0] = deck
            if load_anims:
                self.play(*load_anims, run_time=0.6)

            # 2) all stages compute AT THE SAME TIME; gantt cells appear
            pulse_anims, gantt_fills = [], []
            for c in tick_computes:
                deck = self.stage_deck[c.stage]
                pulse_anims.append(Indicate(deck, scale_factor=1.08, color=style.ACCENT))
                for cell in self.weights[f"stage{c.stage}"]:
                    pulse_anims.append(Indicate(cell, scale_factor=1.04, color=style.WEIGHT_STROKE))
                gantt_fills.append(FadeIn(self.gantt_fill(c.stage, t, c.microbatch)))
            if len(tick_computes) == 2 and t == 1:
                self.show_caption_text(
                    "The overlap: stage 0 starts mb1 while stage 1 processes mb0"
                )
            self.play(*pulse_anims, *gantt_fills, run_time=1.1)

            # 3) a finished last-stage microbatch retires to the output column
            for c in tick_computes:
                if c.stage == len(self.grid) - 1:
                    deck = self.stage_deck[c.stage]
                    self.stage_deck[c.stage] = None
                    target = self.grid.box(c.stage).box.get_right() + RIGHT * 1.1
                    target[1] = 1.9 - 0.85 * len(self.done)
                    self.done.append(deck)
                    self.play(deck.animate.scale(0.8).move_to(target), run_time=0.5)

            # 4) point-to-point handoff to the next stage
            for p in tick_sends:
                deck = self.stage_deck[p.src_stage]
                self.stage_deck[p.src_stage] = None
                self.play(self.strip.show(p.tex(), color=style.COMM_COLOR), run_time=0.35)
                slot = self.grid.box(p.dst_stage).slot_center("acts")
                self.play(deck.animate(path_arc=-0.4).move_to(slot), run_time=0.7)
                self.stage_deck[p.dst_stage] = deck
                self.comm_bump()

    def summary_row(self, op: str, n: int) -> str:
        return f"{op}  ×{n}   (one per microbatch per stage boundary)"

    def bubble_beat(self):
        pp = self.cfg.pipeline
        n_ticks = pp.n_microbatches + pp.n_stages - 1
        empties = []
        for s in range(pp.n_stages):
            for t in range(n_ticks):
                if not (0 <= t - s < pp.n_microbatches):
                    r = Rectangle(width=CELL_W - 0.06, height=CELL_H - 0.06, stroke_width=1.8)
                    r.set_stroke(style.COMM_COLOR)
                    r.move_to(self.gantt_cell_center(s, t))
                    r.set_z_index(style.Z_LABEL + 1)
                    empties.append(DashedVMobject(r, num_dashes=16))
        tag = Text("bubble = idle time; more microbatches → smaller fraction",
                   font_size=18, color=style.COMM_COLOR)
        tag.next_to(self.gantt, UP, buff=0.15)
        tag.set_z_index(style.Z_LABEL + 1)
        self.show_caption_text("Communication: only activation handoffs between adjacent stages")
        self.play(*[FadeIn(e) for e in empties], FadeIn(tag), run_time=0.8)
        self.wait(2.0)
