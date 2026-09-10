"""Expert parallelism on the model canvas: tokens become router-colored
squares that crossfly VERTICALLY between lanes at the MoE station — dispatch
in one column, combine in another, so the station reads "in one side, out the
other".

Token bookkeeping: TOKENS_PER_DEVICE tokens per lane, token k routed to expert
k % n_experts — perfectly balanced (capacity/load-balance are out of scope and
the summary says so).
"""

from __future__ import annotations

import numpy as np
from manim import FadeIn, ReplacementTransform, Square, Text, VGroup

from tpviz import configs, style
from tpviz.core.steps import AllToAllStep, AnnotateStep, RouteStep
from tpviz.scenes.base import ForwardPassScene

TOKENS_PER_DEVICE = 8  # 4 x 2 grid
COLS, ROWS = 4, 2
TOKEN_SIDE = 0.18
TOKEN_PITCH = 0.24


class EPScene(ForwardPassScene):
    cfg = configs.EP

    def place_initial_tensors(self):
        super().place_initial_tensors()
        chips = []
        for s in self.canvas.stations:
            if s.kind == "block" and s.phase == "moe":
                x0, _ = self.canvas.station_span(s.key)
                for i, lane in enumerate(self.canvas.lanes):
                    chip = Text(
                        f"E{i}", font_size=14, weight="BOLD",
                        color=style.EXPERT_HUES[i % len(style.EXPERT_HUES)],
                    )
                    chip.move_to(np.array([x0 + 0.22, lane.weight_y(), 0.0]))
                    chip.set_z_index(style.Z_LABEL)
                    chips.append(chip)
        self.play(*[FadeIn(c) for c in chips], run_time=self.rt(0.4))

    # ---------------------------------------------------------------- tokens
    def _token_grid_positions(self, lane: int, x: float) -> list[np.ndarray]:
        cy = self.canvas.lanes[lane].act_y()
        pts = []
        for r in range(ROWS):
            for c in range(COLS):
                pts.append(
                    np.array(
                        [
                            x + (c - (COLS - 1) / 2) * TOKEN_PITCH,
                            cy + ((ROWS - 1) / 2 - r) * TOKEN_PITCH,
                            0.0,
                        ]
                    )
                )
        return pts

    def _make_token(self, expert: int) -> Square:
        sq = Square(side_length=TOKEN_SIDE, stroke_width=1.2)
        sq.set_fill(style.EXPERT_HUES[expert % len(style.EXPERT_HUES)], opacity=0.95)
        sq.set_stroke("#e5e7eb", opacity=0.6)
        sq.set_z_index(style.Z_TENSOR + 1)
        return sq

    def handle_RouteStep(self, step: RouteStep):
        key = self.canvas.station_key(step.layer, step.phase)
        self._moe_key = key
        self.play(self.strip.show(step.tex(), color=style.ACCENT), run_time=self.rt(0.4))
        n_exp = step.n_experts
        decks = self.acts.pop(step.tokens.name)
        self.tokens: list[list[Square]] = []
        self.token_expert: list[list[int]] = []
        anims = []
        for i in range(self.n_lanes()):
            x = self.canvas.anchor(key, "tokens", i)[0]
            pts = self._token_grid_positions(i, x)
            device_tokens, experts = [], []
            for k in range(TOKENS_PER_DEVICE):
                e = k % n_exp
                tok = self._make_token(e).move_to(pts[k])
                device_tokens.append(tok)
                experts.append(e)
            self.tokens.append(device_tokens)
            self.token_expert.append(experts)
            anims.append(ReplacementTransform(decks[i], VGroup(*device_tokens)))
        self.play(*anims, run_time=self.rt(0.9))
        self.wait(self.rt(0.4))

    def handle_AllToAllStep(self, step: AllToAllStep):
        self.play(self.strip.show(step.tex(), color=style.COMM_COLOR), run_time=self.rt(0.45))
        if step.direction == "dispatch":
            self._dispatch(step)
        else:
            self._combine(step)
        self.comm_bump()

    def _dispatch(self, step: AllToAllStep):
        """Tokens crossfly vertically in the dispatch column, sorting by hue."""
        n = self.n_lanes()
        key = self._moe_key
        arrival_pts = [
            self._token_grid_positions(e, self.canvas.anchor(key, "tokens", e)[0])
            for e in range(n)
        ]
        arrival_count = [0] * n
        arrived: list[list[Square]] = [[] for _ in range(n)]
        flights = []
        for i in range(n):
            for tok, e in zip(self.tokens[i], self.token_expert[i]):
                dest = arrival_pts[e][arrival_count[e] % TOKENS_PER_DEVICE]
                arrival_count[e] += 1
                arrived[e].append(tok)
                tok.set_z_index(style.Z_FLYING)
                flights.append(tok.animate(path_arc=0.3).move_to(dest))
        self.play(*flights, run_time=self.rt(1.7))
        # arrived tokens condense into each expert's routed batch at `core`
        outs = [
            self.build_act_vis(step.out, e, at=self.canvas.anchor(key, "core", e))
            for e in range(n)
        ]
        self.play(
            *[ReplacementTransform(VGroup(*arrived[e]), outs[e]) for e in range(n)],
            run_time=self.rt(0.7),
        )
        self.acts[step.out.name] = outs
        self.set_tracker(step.out, self.canvas.anchor(key, "core", 0)[0])

    def _combine(self, step: AllToAllStep):
        """Processed batches explode in the combine column; tokens fly home."""
        n = self.n_lanes()
        key = self._moe_key
        routed = self.acts.pop(step.src.name)
        home_tokens: list[list[Square]] = [[] for _ in range(n)]
        explode_anims = []
        for e in range(n):
            x = self.canvas.anchor(key, "combine", e)[0]
            pts = self._token_grid_positions(e, x)
            device_tokens = []
            k = 0
            for i in range(n):
                for _ in range(TOKENS_PER_DEVICE // n):
                    tok = self._make_token(e).move_to(pts[k])
                    tok.set_z_index(style.Z_FLYING)
                    device_tokens.append((tok, i))
                    k += 1
            explode_anims.append(
                ReplacementTransform(routed[e], VGroup(*[t for t, _ in device_tokens]))
            )
            for tok, origin in device_tokens:
                home_tokens[origin].append(tok)
        self.play(*explode_anims, run_time=self.rt(0.6))
        flights = []
        for origin in range(n):
            x = self.canvas.anchor(key, "combine", origin)[0]
            pts = self._token_grid_positions(origin, x)
            for k, tok in enumerate(home_tokens[origin]):
                flights.append(tok.animate(path_arc=-0.3).move_to(pts[k % len(pts)]))
        self.play(*flights, run_time=self.rt(1.7))
        outs = [
            self.build_act_vis(step.out, i, at=self.canvas.anchor(key, "exit", i))
            for i in range(n)
        ]
        self.play(
            *[ReplacementTransform(VGroup(*home_tokens[i]), outs[i]) for i in range(n)],
            run_time=self.rt(0.7),
        )
        self.acts[step.out.name] = outs
        self.set_tracker(step.out, self.canvas.anchor(key, "exit", 0)[0])

    def summary(self):
        self.speed = 1.0
        self.play_step(
            AnnotateStep(
                text="Two AllToAlls per MoE layer. (Capacity & load-balancing ignored here.)",
                layer=2,
                phase="moe",
            )
        )
        super().summary()
