"""Expert parallelism on the model canvas: tokens become router-colored
squares that crossfly VERTICALLY between lanes at the MoE station — dispatch
in one column, combine in another, so the station reads "in one side, out the
other". The backward pass mirrors it: token GRADIENTS AllToAll to their
experts (entering from the right) and home again (leaving left).

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


class MoETokenMixin:
    """Token-level AllToAll choreography shared by the EP forward-only and
    forward+backward scenes. Mixed into a ForwardPassScene subclass."""

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

    def _make_token(self, expert: int, *, grad: bool = False) -> Square:
        sq = Square(side_length=TOKEN_SIDE, stroke_width=1.6 if grad else 1.2)
        sq.set_fill(style.EXPERT_HUES[expert % len(style.EXPERT_HUES)],
                    opacity=0.7 if grad else 0.95)
        sq.set_stroke(style.GRAD_STROKE if grad else "#e5e7eb", opacity=0.9 if grad else 0.6)
        sq.set_z_index(style.Z_TENSOR + 1)
        return sq

    def _decks_to_tokens(self, decks: list, key: str, column: str, *, grad: bool
                         ) -> tuple[list[list[Square]], list[list[int]]]:
        """Each lane's deck explodes into its token grid at `column`."""
        n_exp = self.cfg.moe.n_experts
        tokens, experts, anims = [], [], []
        for i in range(self.n_lanes()):
            x = self.canvas.anchor(key, column, i)[0]
            pts = self._token_grid_positions(i, x)
            device_tokens, device_experts = [], []
            for k in range(TOKENS_PER_DEVICE):
                e = k % n_exp
                device_tokens.append(self._make_token(e, grad=grad).move_to(pts[k]))
                device_experts.append(e)
            tokens.append(device_tokens)
            experts.append(device_experts)
            anims.append(ReplacementTransform(decks[i], VGroup(*device_tokens)))
        self.play(*anims, run_time=self.rt(0.9))
        return tokens, experts

    def handle_RouteStep(self, step: RouteStep):
        key = self.canvas.station_key(step.layer, step.phase)
        self._moe_key = key
        self.play(self.strip.show(step.tex(), color=style.ACCENT), run_time=self.rt(0.4))
        decks = self.acts.pop(step.tokens.name)
        self.tokens, self.token_expert = self._decks_to_tokens(decks, key, "tokens", grad=False)
        self.wait(self.rt(0.4))

    def handle_AllToAllStep(self, step: AllToAllStep):
        self.play(self.strip.show(step.tex(), color=style.COMM_COLOR), run_time=self.rt(0.45))
        key = self.canvas.station_key(step.layer, step.phase)
        self._moe_key = key
        if not step.backward:
            if step.direction == "dispatch":
                self._crossfly_to_experts(step, key, land="core")
            else:
                self._crossfly_home(step, key, explode_at="combine", land="exit")
        else:
            # mirrored: gradients enter from the right, leave to the left
            if step.direction == "dispatch":
                decks = self.acts.pop(step.src.name)
                self.tokens, self.token_expert = self._decks_to_tokens(
                    decks, key, "combine", grad=True
                )
                self._crossfly_to_experts(step, key, land="core", grad=True,
                                          column="combine")
            else:
                self._crossfly_home(step, key, explode_at="tokens", land="entry", grad=True)
        self.comm_bump()

    def _crossfly_to_experts(self, step: AllToAllStep, key: str, *, land: str,
                             grad: bool = False, column: str = "tokens"):
        """token (i, k) flies vertically to lane e = expert(k); arrivals condense."""
        n = self.n_lanes()
        arrival_pts = [
            self._token_grid_positions(e, self.canvas.anchor(key, column, e)[0])
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
        outs = [
            self.build_act_vis(step.out, e, at=self.canvas.anchor(key, land, e))
            for e in range(n)
        ]
        self.play(
            *[ReplacementTransform(VGroup(*arrived[e]), outs[e]) for e in range(n)],
            run_time=self.rt(0.7),
        )
        self.acts[step.out.name] = outs
        self.set_tracker(step.out, self.canvas.anchor(key, land, 0)[0])

    def _crossfly_home(self, step: AllToAllStep, key: str, *, explode_at: str,
                       land: str, grad: bool = False):
        """expert batches explode; tokens fly home vertically; condense per lane."""
        n = self.n_lanes()
        routed = self.acts.pop(step.src.name)
        home_tokens: list[list[Square]] = [[] for _ in range(n)]
        explode_anims = []
        for e in range(n):
            x = self.canvas.anchor(key, explode_at, e)[0]
            pts = self._token_grid_positions(e, x)
            device_tokens = []
            k = 0
            for i in range(n):
                for _ in range(TOKENS_PER_DEVICE // n):
                    tok = self._make_token(e, grad=grad).move_to(pts[k])
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
            x = self.canvas.anchor(key, explode_at, origin)[0]
            pts = self._token_grid_positions(origin, x)
            for k, tok in enumerate(home_tokens[origin]):
                flights.append(tok.animate(path_arc=-0.3).move_to(pts[k % len(pts)]))
        self.play(*flights, run_time=self.rt(1.7))
        outs = [
            self.build_act_vis(step.out, i, at=self.canvas.anchor(key, land, i))
            for i in range(n)
        ]
        self.play(
            *[ReplacementTransform(VGroup(*home_tokens[i]), outs[i]) for i in range(n)],
            run_time=self.rt(0.7),
        )
        self.acts[step.out.name] = outs
        self.set_tracker(step.out, self.canvas.anchor(key, land, 0)[0])


class EPScene(MoETokenMixin, ForwardPassScene):
    cfg = configs.EP

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
