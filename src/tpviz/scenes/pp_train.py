"""Pipeline parallelism, forward + backward, on the shared canvas.

Forward drains as in the PP scene; then microbatch GRADIENTS (rose) flow
back right -> left, hopping UP at the stage boundary. Backward Gantt cells
are TWO ticks wide — the backward pass costs ~2x the forward — which is why
the pipeline bubble grows during training.
"""

from __future__ import annotations

import numpy as np
from manim import DOWN, FadeIn, Indicate, Rectangle, ReplacementTransform, Text, VGroup

from tpviz import configs, style
from tpviz.core.backward import train_steps
from tpviz.core.steps import P2PSendStep, StageComputeStep
from tpviz.core.tensors import LTensor
from tpviz.mobjects.labels import caption_text
from tpviz.mobjects.tensor_mobject import ActivationDeck
from tpviz.scenes.base import PHASE_WEIGHTS
from tpviz.scenes.pp import PPScene


class PPTrainScene(PPScene):
    cfg = configs.PP
    cell_w = 0.42
    cell_h = 0.32

    def gantt_n_ticks(self) -> int:
        pp = self.cfg.pipeline
        f_end = pp.n_microbatches + pp.n_stages - 1
        return f_end + 2 * (pp.n_microbatches + pp.n_stages - 1)

    def gantt_fill_bwd(self, stage: int, tick: int, mb: int) -> VGroup:
        """A double-width, rose-bordered cell: backward costs 2x."""
        c = Rectangle(width=2 * self.cell_w - 0.06, height=self.cell_h - 0.06, stroke_width=1.6)
        c.set_fill(style.MICROBATCH_HUES[mb % len(style.MICROBATCH_HUES)], opacity=0.55)
        c.set_stroke(style.GRAD_STROKE, opacity=0.95)
        c.move_to(
            (self.gantt_cell_center(stage, tick) + self.gantt_cell_center(stage, tick + 1)) / 2
        )
        c.set_z_index(style.Z_LABEL + 1)
        return c

    def run_pipeline(self):
        super().run_pipeline()  # the forward drain (plays only non-backward steps)
        self.run_backward_pipeline()

    def run_backward_pipeline(self):
        steps = train_steps(self.cfg)
        computes = [s for s in steps if isinstance(s, StageComputeStep) and s.backward]
        sends = [s for s in steps if isinstance(s, P2PSendStep) and s.backward]
        ticks = sorted({c.tick for c in computes})

        self.show_caption_text(
            "Backward: gradients flow right → left; every backward cell costs 2×"
        )
        grad_t = LTensor("dX", ("B", "T", "D"), self.cfg.act_sharding, kind="grad")
        last = self.n_lanes() - 1
        returned = 0

        for t in ticks:
            tick_computes = [c for c in computes if c.tick == t]
            tick_sends = [p for p in sends if p.tick == t + 1]

            lead = tick_computes[0]
            lead.caption = " · ".join(
                f"stage {c.stage} backprops mb{c.microbatch}" for c in tick_computes
            )
            self.mark_step(lead)

            # 1) the last stage picks its next finished microbatch from the Out
            #    dock — its activations become a GRADIENT deck (rose)
            for c in tick_computes:
                if c.stage == last:
                    deck = self.done_by_mb[c.microbatch]
                    hue = style.MICROBATCH_HUES[c.microbatch % len(style.MICROBATCH_HUES)]
                    rose = ActivationDeck(grad_t, self.cfg.mesh, scale=0.34)
                    from manim import SurroundingRectangle

                    frame = SurroundingRectangle(rose, color=hue, stroke_width=2.2, buff=0.05)
                    tag = Text(f"dX mb{c.microbatch}", font_size=12, color=style.GRAD_STROKE,
                               weight="BOLD")
                    tag.next_to(frame, DOWN, buff=0.05)
                    g = VGroup(rose, frame, tag)
                    g.move_to(self.canvas.anchor(self._mlp_key(last), "exit", last))
                    g.set_z_index(style.Z_TENSOR)
                    self.play(ReplacementTransform(deck, g), run_time=0.5)
                    self.stage_deck[last] = g

            # 2) all active stages traverse their stations RIGHT -> LEFT together
            for key_name in ("_mlp_key", "_attn_key"):
                anims, fills = [], []
                for c in tick_computes:
                    deck = self.stage_deck[c.stage]
                    key = getattr(self, key_name)(c.stage)
                    anims.append(
                        deck.animate.move_to(self.canvas.anchor(key, "entry", c.stage))
                    )
                    for name in PHASE_WEIGHTS[self.canvas.station(key).phase]:
                        for vis in self.weights[(key, name)]:
                            anims.append(
                                Indicate(vis, scale_factor=1.05, color=style.GRAD_STROKE)
                            )
                    if key_name == "_mlp_key":
                        fills.append(FadeIn(self.gantt_fill_bwd(c.stage, t, c.microbatch)))
                self.play(*anims, *fills, run_time=0.9)

            # 3) a gradient finishing stage 0 retires into the In dock (dIn)
            for c in tick_computes:
                if c.stage == 0:
                    deck = self.stage_deck[0]
                    self.stage_deck[0] = None
                    self.play(
                        deck.animate.move_to(
                            self.canvas.dock_slot("in", returned, pitch=0.92)
                        ),
                        run_time=0.45,
                    )
                    returned += 1

            # 4) P2P: the gradient hops UP-LEFT across the stage boundary
            for p in tick_sends:
                self.flush_mark()
                self.mark_step(p)
                deck = self.stage_deck[p.src_stage]
                self.stage_deck[p.src_stage] = None
                self.play(self.strip.show(p.tex(), color=style.GRAD_STROKE, note=p.note_tex), run_time=0.3)
                self.play(
                    deck.animate(path_arc=-0.35).move_to(
                        self.canvas.anchor(self._mlp_key(p.dst_stage), "exit", p.dst_stage)
                    ),
                    run_time=0.65,
                )
                self.stage_deck[p.dst_stage] = deck
                self.comm_bump()
            self.flush_mark()

    def bubble_beat(self):
        tag = Text(
            "backward cells are 2× wide — the bubble grows with the backward pass",
            font_size=17,
            color=style.GRAD_STROKE,
        )
        tag.next_to(self.gantt, np.array([0.0, 1.0, 0.0]), buff=0.1)
        tag.set_z_index(style.Z_LABEL + 1)
        self.show_caption_text(
            "Communication: activation handoffs forward, gradient handoffs backward"
        )
        self.play(FadeIn(tag), run_time=0.6)
        self.wait(2.0)

    def summary_steps(self):
        return train_steps(self.cfg)

    def dock_out_finale(self):
        pass  # gradient decks already retired into the In dock

    def summary_row(self, op: str, n: int) -> str:
        return f"{op}  ×{n}   (forward sends + backward sends)"
