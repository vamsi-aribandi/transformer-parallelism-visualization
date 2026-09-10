"""Expert parallelism: tokens become router-colored squares that crossfly.

Token bookkeeping: TOKENS_PER_DEVICE tokens per device, token k routed to
expert k % n_experts — perfectly balanced, so every device sends and receives
the same number (capacity/load-balance issues are out of scope and said so).
"""

from __future__ import annotations

import numpy as np
from manim import DOWN, FadeIn, ReplacementTransform, Square, Text, VGroup

from tpviz import configs, style
from tpviz.core.steps import AllToAllStep, RouteStep
from tpviz.scenes.base import ForwardPassScene

TOKENS_PER_DEVICE = 8  # 4 x 2 grid
COLS, ROWS = 4, 2


class EPScene(ForwardPassScene):
    cfg = configs.EP

    def place_initial_tensors(self):
        super().place_initial_tensors()
        tags = []
        for i, box in enumerate(self.grid.boxes):
            tag = Text(
                f"Expert {i}", font_size=16, weight="BOLD",
                color=style.EXPERT_HUES[i % len(style.EXPERT_HUES)],
            )
            tag.next_to(box.title, DOWN, buff=0.06)
            tag.set_z_index(style.Z_LABEL)
            tags.append(tag)
        self.play(*[FadeIn(t) for t in tags], run_time=self.rt(0.5))

    # ---------------------------------------------------------------- tokens
    def _token_grid_positions(self, device: int) -> list[np.ndarray]:
        center = self.grid.box(device).slot_center("acts")
        dx, dy = 0.42, 0.42
        pts = []
        for r in range(ROWS):
            for c in range(COLS):
                pts.append(
                    center
                    + np.array(
                        [(c - (COLS - 1) / 2) * dx, ((ROWS - 1) / 2 - r) * dy, 0.0]
                    )
                )
        return pts

    def _make_token(self, expert: int) -> Square:
        sq = Square(side_length=0.3, stroke_width=1.5)
        sq.set_fill(style.EXPERT_HUES[expert % len(style.EXPERT_HUES)], opacity=0.95)
        sq.set_stroke("#e5e7eb", opacity=0.6)
        sq.set_z_index(style.Z_TENSOR + 1)
        return sq

    def handle_RouteStep(self, step: RouteStep):
        self.play(self.strip.show(step.tex(), color=style.ACCENT), run_time=self.rt(0.4))
        n_exp = step.n_experts
        decks = self.acts.pop(step.tokens.name)
        self.token_src_name = step.tokens.name
        self.tokens: list[list[Square]] = []
        self.token_expert: list[list[int]] = []
        anims = []
        for i in range(len(self.grid)):
            pts = self._token_grid_positions(i)
            device_tokens, experts = [], []
            for k in range(TOKENS_PER_DEVICE):
                e = k % n_exp
                tok = self._make_token(e).move_to(pts[k])
                device_tokens.append(tok)
                experts.append(e)
            self.tokens.append(device_tokens)
            self.token_expert.append(experts)
            anims.append(ReplacementTransform(decks[i], VGroup(*device_tokens)))
        self.play(*anims, run_time=self.rt(1.0))
        self.wait(self.rt(0.5))

    def handle_AllToAllStep(self, step: AllToAllStep):
        self.play(self.strip.show(step.tex(), color=style.COMM_COLOR), run_time=self.rt(0.5))
        if step.direction == "dispatch":
            self._dispatch(step)
        else:
            self._combine(step)
        self.comm_bump()

    def _dispatch(self, step: AllToAllStep):
        n = len(self.grid)
        # arrival slot bookkeeping: tokens arriving at device e stack in order
        arrival_pts = [self._token_grid_positions(e) for e in range(n)]
        arrival_count = [0] * n
        arrived: list[list[Square]] = [[] for _ in range(n)]
        flights = []
        for i in range(n):
            for tok, e in zip(self.tokens[i], self.token_expert[i]):
                dest = arrival_pts[e][arrival_count[e] % TOKENS_PER_DEVICE]
                arrival_count[e] += 1
                arrived[e].append(tok)
                tok.set_z_index(style.Z_FLYING)
                flights.append(tok.animate(path_arc=0.4).move_to(dest))
        self.play(*flights, run_time=self.rt(1.8))
        # arrived tokens condense into each expert's routed batch
        outs = [self.build_act_vis(step.out, e) for e in range(n)]
        self.play(
            *[ReplacementTransform(VGroup(*arrived[e]), outs[e]) for e in range(n)],
            run_time=self.rt(0.8),
        )
        self.acts[step.out.name] = outs

    def _combine(self, step: AllToAllStep):
        n = len(self.grid)
        routed = self.acts.pop(step.src.name)
        # explode each expert's processed batch back into its tokens
        flights = []
        home_tokens: list[list[Square]] = [[] for _ in range(n)]
        explode_anims = []
        for e in range(n):
            device_tokens = []
            pts = self._token_grid_positions(e)
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
        self.play(*explode_anims, run_time=self.rt(0.7))
        for origin in range(n):
            pts = self._token_grid_positions(origin)
            for k, tok in enumerate(home_tokens[origin]):
                flights.append(tok.animate(path_arc=-0.4).move_to(pts[k % len(pts)]))
        self.play(*flights, run_time=self.rt(1.8))
        outs = [self.build_act_vis(step.out, i) for i in range(n)]
        self.play(
            *[ReplacementTransform(VGroup(*home_tokens[i]), outs[i]) for i in range(n)],
            run_time=self.rt(0.8),
        )
        self.acts[step.out.name] = outs

    def summary(self):
        self.speed = 1.0
        from tpviz.core.steps import AnnotateStep

        self.play_step(
            AnnotateStep(
                text="Two AllToAlls per MoE layer. (Capacity & load-balancing ignored here.)",
                layer=2,
                phase="moe",
            )
        )
        super().summary()
