"""ForwardPassScene: plays a StrategyConfig's forward-pass step list on the
horizontal ModelCanvas — model depth runs left -> right (stations), devices are
stacked lanes, collectives fly vertically between lanes at the current x.

Each concrete scene is a thin subclass setting `cfg`. Handlers are dispatched
by step class name (`handle_MatMulStep`, ...); pipeline-specific steps are
handled by the PP scene subclass.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    FadeIn,
    FadeOut,
    FadeTransform,
    Indicate,
    LaggedStart,
    ReplacementTransform,
    Scene,
    Text,
    TransformFromCopy,
    VGroup,
)

from tpviz import style
from tpviz.animations import collectives as coll
from tpviz.animations import compute
from tpviz.core.mesh import StrategyConfig
from tpviz.core.model import count_collectives, forward_steps, make_input, make_weight
from tpviz.core.steps import (
    AllGatherStep,
    AllReduceStep,
    AllToAllStep,
    AnnotateStep,
    AttentionCoreStep,
    GeluStep,
    MatMulStep,
    ReduceScatterStep,
    RouteStep,
    SplitQKVStep,
    Step,
)
from tpviz.core.tensors import LTensor
from tpviz.mobjects.canvas import DECK_SCALE, FIXTURE_SCALE, MINI_SCALE, ModelCanvas
from tpviz.mobjects.equation_strip import EquationStrip
from tpviz.mobjects.labels import caption_text, math_label, tensor_label
from tpviz.mobjects.tensor_mobject import ActivationDeck, WeightRect

PHASE_LABEL = {"attn": "Attention", "mlp": "MLP", "moe": "MoE MLP", "pipeline": "Pipeline"}
LAYER2_SPEED = 0.45
CAPTION_Y = 3.1
TRACKER_Y = -2.87

# which weights live at each phase's fixture slots, and where a matmul's
# product lands (the rightward drift *is* the ReplacementTransform target)
PHASE_WEIGHTS = {"attn": ("W_qkv", "W_o"), "mlp": ("W_in", "W_out"), "moe": ("W_in", "W_out")}
NEXT_ANCHOR = {
    ("attn", 0): "core",
    ("attn", 1): "exit",
    ("mlp", 0): "mid",
    ("mlp", 1): "exit",
    ("moe", 0): "core",
    ("moe", 1): "combine",
}


def default_caption(step: Step) -> str | None:
    if isinstance(step, AllGatherStep):
        if step.jit:
            return "Gather the full weight just in time — use it, then discard it"
        if step.src.kind == "kv":
            return "Attention needs every token: gather K and V across shards"
        return "Gather the sharded activations before the matmul"
    if isinstance(step, ReduceScatterStep):
        return "Unreduced partial sums resolve — each device keeps one slice"
    if isinstance(step, AllReduceStep):
        return "Unreduced partial sums resolve everywhere"
    if isinstance(step, AllToAllStep):
        if step.direction == "dispatch":
            return "AllToAll: every token travels to its expert's device"
        return "AllToAll: processed tokens return home"
    if isinstance(step, RouteStep):
        return "The router assigns each token to an expert"
    return None


class ForwardPassScene(Scene):
    cfg: ClassVar[StrategyConfig]

    # ------------------------------------------------------------- lifecycle
    def construct(self):
        self.speed = 1.0
        self.comm_count = 0
        self.acts: dict[str, list[VGroup]] = {}
        self.weights: dict[tuple[str, str], list[VGroup]] = {}
        self.pending_restore: dict[tuple[str, str], list[VGroup]] = {}
        self.caption_mobj: Text | None = None
        self.badge: Text | None = None
        self.tracker = None
        self.highlight = None
        self.cur_station: str | None = None

        self.next_section("title")
        self.title_card()
        self.next_section("setup")
        self.setup_stage()
        self.place_initial_tensors()
        cur = (None, None)
        for step in forward_steps(self.cfg):
            if (step.layer, step.phase) != cur:
                cur = (step.layer, step.phase)
                self.next_section(f"l{step.layer}-{step.phase}")
                self.speed = 1.0 if step.layer <= 1 else LAYER2_SPEED
                self.on_phase(step)
            self.speed = 1.0 if step.layer <= 1 else LAYER2_SPEED
            self.play_step(step)
        self.next_section("summary")
        self.summary()

    def rt(self, base: float) -> float:
        return base * self.speed

    def n_lanes(self) -> int:
        return len(self.canvas.lanes)

    # ----------------------------------------------------------------- intro
    def title_card(self):
        name = Text(self.cfg.name, font_size=44, color=style.TEXT_COLOR, weight="BOLD")
        tag = Text(self.cfg.tagline, font_size=26, color=style.MUTED_TEXT)
        mesh = math_label(self.cfg.mesh.tex(), font_size=34, color=style.ACCENT)
        card = VGroup(name, tag, mesh).arrange(DOWN, buff=0.45)
        self.play(FadeIn(card, shift=UP * 0.3), run_time=1.0)
        self.wait(1.4)

        header = VGroup(
            Text(self.cfg.name, font_size=22, color=style.TEXT_COLOR, weight="BOLD"),
            math_label(self.cfg.mesh.tex(), font_size=22, color=style.ACCENT),
        ).arrange(RIGHT, buff=0.4)
        header.to_corner(UP + LEFT, buff=0.2)
        header.set_z_index(style.Z_LABEL)
        self.play(ReplacementTransform(card, header), run_time=0.8)
        self.header = header

    def station_header_tex(self, key: str, override: LTensor | None = None) -> str:
        """The once-per-station weight notation line, e.g. `W_qkv[D,H_Y]  W_o[H_Y,D]`."""
        s = self.canvas.station(key)
        parts = []
        for name in PHASE_WEIGHTS[s.phase]:
            t = make_weight(self.cfg, name)
            if override is not None and override.name == name:
                t = override
            parts.append(t.tex())
        return r"\quad ".join(parts)

    def make_canvas(self) -> ModelCanvas:
        return ModelCanvas(self.cfg)

    def setup_stage(self):
        self.canvas = self.make_canvas()
        for s in self.canvas.stations:
            if s.kind == "block":
                self.canvas.add_header_tex(s.key, self.station_header_tex(s.key))
        self.strip = EquationStrip()
        self.add(self.strip)

        self.comm_label = caption_text("collectives", font_size=16, color=style.MUTED_TEXT)
        self.comm_value = caption_text("0", font_size=30, color=style.GOOD_COLOR)
        counter = VGroup(self.comm_label, self.comm_value).arrange(DOWN, buff=0.08)
        counter.to_corner(UP + RIGHT, buff=0.2)
        self.add(counter)

        self.play(FadeIn(self.canvas), run_time=1.0)

    def place_initial_tensors(self):
        cfg = self.cfg
        # weight fixtures at every block station, every lane
        fixtures = []
        for s in self.canvas.stations:
            if s.kind != "block":
                continue
            for slot, name in enumerate(PHASE_WEIGHTS[s.phase]):
                t = make_weight(cfg, name)
                per_lane = []
                for i in range(self.n_lanes()):
                    vis = WeightRect(t, cfg.mesh, device=i, scale=FIXTURE_SCALE)
                    self.canvas.fit_weight(VGroup(vis), s.key, slot, i)
                    per_lane.append(vis)
                    fixtures.append(vis)
                self.weights[(s.key, name)] = per_lane

        # the input batch: one full deck splits into per-lane shards at the In dock
        t_in = make_input(cfg)
        full = ActivationDeck(t_in, cfg.mesh, sharded=False, scale=0.8)
        full_label = full.make_label(font_size=26)
        full_group = VGroup(full, full_label)
        full_group.move_to(UP * 0.2)
        full_group.set_z_index(style.Z_FLYING)

        shards = []
        for i in range(self.n_lanes()):
            vis = ActivationDeck(t_in, cfg.mesh, device=i, scale=DECK_SCALE)
            self.canvas.fit_act(VGroup(vis), "in", "dock", i)
            shards.append(VGroup(vis))
        self.acts["In"] = shards

        self.show_caption_text("Place the tensors: " + self.cfg.tagline)
        self.play(
            LaggedStart(*[FadeIn(f) for f in fixtures], lag_ratio=0.01),
            run_time=self.rt(1.4),
        )
        self.play(FadeIn(full_group, scale=1.1), run_time=self.rt(0.7))
        self.wait(self.rt(0.5))
        self.play(
            *[TransformFromCopy(full, g) for g in shards],
            FadeOut(full_group),
            run_time=self.rt(1.1),
        )
        self.set_tracker(t_in, self.canvas.anchor("in", "dock", 0)[0])

    # ------------------------------------------------------------- utilities
    def build_act_vis(
        self,
        t: LTensor,
        lane: int,
        *,
        at: np.ndarray | None = None,
        like: VGroup | None = None,
        scale: float = DECK_SCALE,
    ) -> VGroup:
        vis = ActivationDeck(t, self.cfg.mesh, device=lane, scale=scale)
        g = VGroup(vis)
        if like is not None:
            s = min(1.0, like.width / max(g.width, 1e-6), like.height / max(g.height, 1e-6))
            g.scale(max(s, 0.35))
            g.move_to(like.get_center())
        elif at is not None:
            lane_h = self.canvas.lanes[lane].rect.height
            s = min(1.0, 0.95 / max(g.width, 1e-6), lane_h * 0.52 / max(g.height, 1e-6))
            g.scale(s)
            g.move_to(at)
        return g

    def set_tracker(self, t: LTensor, x: float, run_time: float | None = None):
        """The single shared traveling notation label under the canvas."""
        new = tensor_label(t, font_size=26)
        if new.height > 0.38:
            new.scale(0.38 / new.height)
        x = min(max(x, -7.0 + new.width / 2 + 0.1), 7.0 - new.width / 2 - 0.1)
        new.move_to(np.array([x, TRACKER_Y, 0.0]))
        if self.tracker is None:
            self.tracker = new
            self.play(FadeIn(new), run_time=run_time or self.rt(0.25))
        else:
            old, self.tracker = self.tracker, new
            self.play(FadeTransform(old, new), run_time=run_time or self.rt(0.25))

    def show_caption_text(self, text: str | None):
        if text is None or self.speed < 1.0:
            return
        cap = caption_text(text, font_size=23)
        if cap.width > 12.8:
            cap.scale(12.8 / cap.width)
        cap.move_to(UP * CAPTION_Y)
        if self.caption_mobj is None:
            self.play(FadeIn(cap), run_time=0.4)
        else:
            self.play(ReplacementTransform(self.caption_mobj, cap), run_time=0.4)
        self.caption_mobj = cap

    def on_phase(self, step: Step):
        """Advance to the step's station: badge, highlight, decks slide right."""
        key = self.canvas.station_key(step.layer, step.phase)
        self.cur_station = key

        label = f"Layer {step.layer} · {PHASE_LABEL.get(step.phase, step.phase)}"
        badge = Text(label, font_size=20, color=style.ACCENT, weight="BOLD")
        badge.next_to(self.header, DOWN, buff=0.12, aligned_edge=LEFT)
        badge.set_z_index(style.Z_LABEL)

        anims = []
        if self.caption_mobj is not None:
            anims.append(FadeOut(self.caption_mobj))
            self.caption_mobj = None
        if self.badge is None:
            anims.append(FadeIn(badge))
        else:
            anims.append(ReplacementTransform(self.badge, badge))
        self.badge = badge

        hl = self.canvas.station_highlight(key)
        if self.highlight is None:
            anims.append(FadeIn(hl))
            self.highlight = hl
            self.add(hl)
        else:
            anims.append(self.highlight.animate.move_to(hl.get_center()))

        # slide every live activation to the new station's entry
        for vis_list in self.acts.values():
            for i, g in enumerate(vis_list):
                anims.append(g.animate.move_to(self.canvas.anchor(key, "entry", i)))
        self.play(*anims, run_time=self.rt(0.6))

    def comm_bump(self):
        self.comm_count += 1
        new = caption_text(str(self.comm_count), font_size=30, color=style.COMM_COLOR)
        new.move_to(self.comm_value.get_center())
        self.play(ReplacementTransform(self.comm_value, new), run_time=0.3)
        self.comm_value = new

    def play_step(self, step: Step):
        self.show_caption_text(step.caption or default_caption(step))
        getattr(self, f"handle_{type(step).__name__}")(step)

    def phase_of(self, step: Step) -> str:
        return step.phase

    # -------------------------------------------------------------- handlers
    def handle_MatMulStep(self, step: MatMulStep):
        key = self.canvas.station_key(step.layer, step.phase)
        slot = 0 if step.b.name in ("W_qkv", "W_in") else 1
        wkey = (key, step.b.name)
        fixtures = self.weights[wkey]
        acts = self.acts.pop(step.a.name)

        # slide the operand under the weight fixture (MoE computes in place at core)
        anchor_name = "core" if step.phase == "moe" else ("w1" if slot == 0 else "w2")
        self.play(
            *[
                g.animate.move_to(self.canvas.anchor(key, anchor_name, i))
                for i, g in enumerate(acts)
            ],
            self.strip.show(step.tex()),
            run_time=self.rt(0.45),
        )
        out_anchor = NEXT_ANCHOR[(step.phase, slot)]
        outs = [
            self.build_act_vis(step.out, i, at=self.canvas.anchor(key, out_anchor, i))
            for i in range(len(acts))
        ]
        compute.animate_matmul(self, fixtures, acts, outs, run_time=self.rt(1.2))
        self.acts[step.out.name] = outs
        self.set_tracker(step.out, self.canvas.anchor(key, out_anchor, 0)[0])
        self.restore_after_use(key, step.b.name)

    def restore_after_use(self, key: str, weight_name: str):
        """FSDP: the gathered weight was temporary — discard it, shard returns."""
        originals = self.pending_restore.pop((key, weight_name), None)
        if originals is None:
            return
        gathered = self.weights[(key, weight_name)]
        self.play(
            *[FadeOut(g) for g in gathered],
            *[FadeIn(o) for o in originals],
            self.canvas.set_header_weight_tex(key, self.station_header_tex(key)),
            run_time=self.rt(0.6),
        )
        self.weights[(key, weight_name)] = originals

    def handle_SplitQKVStep(self, step: SplitQKVStep):
        key = self.canvas.station_key(step.layer, step.phase)
        src_vis = self.acts.pop(step.src.name)
        per_name: dict[str, list[VGroup]] = {"Q": [], "K": [], "V": []}
        anims = []
        for i in range(self.n_lanes()):
            minis = []
            for t in (step.q, step.k, step.v):
                vis = ActivationDeck(t, self.cfg.mesh, device=i, scale=MINI_SCALE)
                chip = caption_text(t.name, font_size=13, color=style.TENSOR_STROKE[t.kind])
                chip.next_to(vis, DOWN, buff=0.05)
                minis.append(VGroup(vis, chip))
            row = VGroup(*minis).arrange(RIGHT, buff=0.14, aligned_edge=UP)
            lane_h = self.canvas.lanes[i].rect.height
            s = min(1.0, 1.5 / max(row.width, 1e-6), lane_h * 0.56 / max(row.height, 1e-6))
            row.scale(s)
            row.move_to(self.canvas.anchor(key, "core", i))
            for name, m in zip(("Q", "K", "V"), minis):
                per_name[name].append(m)
            anims.append(ReplacementTransform(src_vis[i], row))
        self.play(
            self.strip.show(
                rf"{step.src.tex()} \,\to\, {step.q.tex()},\ {step.k.tex()},\ {step.v.tex()}"
            ),
            run_time=self.rt(0.4),
        )
        self.play(*anims, run_time=self.rt(0.8))
        self.acts.update(per_name)

    def handle_AttentionCoreStep(self, step: AttentionCoreStep):
        key = self.canvas.station_key(step.layer, step.phase)
        qkv_groups = [
            [self.acts["Q"][i], self.acts["K"][i], self.acts["V"][i]]
            for i in range(self.n_lanes())
        ]
        for name in ("Q", "K", "V"):
            self.acts.pop(name)
        capsules, outs = [], []
        for i in range(self.n_lanes()):
            c = compute.make_attn_capsule()
            lane_h = self.canvas.lanes[i].rect.height
            c.scale(min(1.0, lane_h * 0.5 / c.height))
            c.move_to(self.canvas.anchor(key, "core", i))
            capsules.append(c)
            outs.append(
                self.build_act_vis(step.out, i, at=self.canvas.anchor(key, "core", i))
            )
        self.play(self.strip.show(step.tex()), run_time=self.rt(0.4))
        compute.animate_attention(self, qkv_groups, capsules, outs, run_time=self.rt(1.6))
        self.acts[step.out.name] = outs
        self.set_tracker(step.out, self.canvas.anchor(key, "core", 0)[0])

    def handle_GeluStep(self, step: GeluStep):
        self.play(self.strip.show(step.tex()), run_time=self.rt(0.35))
        compute.animate_gelu(self, self.acts[step.src.name], run_time=self.rt(0.6))

    def handle_AllGatherStep(self, step: AllGatherStep):
        key = self.canvas.station_key(step.layer, step.phase)
        self.play(
            self.strip.show(step.tex(), color=style.COMM_COLOR), run_time=self.rt(0.45)
        )
        if step.src.kind == "weight":
            slot = 0 if step.src.name in ("W_qkv", "W_in") else 1
            wkey = (key, step.src.name)
            shards = self.weights[wkey]
            self.pending_restore[wkey] = [s.copy() for s in shards]
            results = []
            for i in range(len(shards)):
                vis = WeightRect(step.out, self.cfg.mesh, device=i, scale=FIXTURE_SCALE)
                self.canvas.fit_weight(VGroup(vis), key, slot, i)
                results.append(vis)
            self.play(
                self.canvas.set_header_weight_tex(
                    key, self.station_header_tex(key, override=step.out)
                ),
                run_time=self.rt(0.3),
            )
            coll.animate_all_gather(self, shards, results, run_time=self.rt(1.4))
            self.weights[wkey] = results
        else:
            shards = self.acts.pop(step.src.name)
            results = [
                self.build_act_vis(step.out, i, like=shards[i]) for i in range(len(shards))
            ]
            coll.animate_all_gather(self, shards, results, run_time=self.rt(1.4))
            self.acts[step.out.name] = results
            self.set_tracker(step.out, results[0].get_center()[0])
        self.comm_bump()

    def handle_ReduceScatterStep(self, step: ReduceScatterStep):
        self._exchange(step)

    def handle_AllReduceStep(self, step: AllReduceStep):
        self._exchange(step)

    def _exchange(self, step):
        key = self.canvas.station_key(step.layer, step.phase)
        self.play(self.strip.show(step.tex(), color=style.COMM_COLOR), run_time=self.rt(0.45))
        partials = self.acts.pop(step.src.name)
        results = [
            self.build_act_vis(step.out, i, at=self.canvas.anchor(key, "exit", i))
            for i in range(len(partials))
        ]
        coll.animate_exchange_resolve(self, partials, results, run_time=self.rt(1.4))
        self.acts[step.out.name] = results
        self.set_tracker(step.out, self.canvas.anchor(key, "exit", 0)[0])
        self.comm_bump()

    def handle_AllToAllStep(self, step: AllToAllStep):
        self._exchange(step)

    def handle_RouteStep(self, step: RouteStep):
        self.play(self.strip.show(step.tex(), color=style.ACCENT), run_time=self.rt(0.35))
        decks = self.acts[step.tokens.name]
        self.play(
            *[
                Indicate(d, scale_factor=1.07, color=style.EXPERT_HUES[i % len(style.EXPERT_HUES)])
                for i, d in enumerate(decks)
            ],
            run_time=self.rt(0.8),
        )

    def handle_AnnotateStep(self, step: AnnotateStep):
        big = caption_text(step.text, font_size=30, color=style.ACCENT)
        if big.width > 12.5:
            big.scale(12.5 / big.width)
        big.move_to(UP * CAPTION_Y)
        old = self.caption_mobj
        if old is not None:
            self.play(ReplacementTransform(old, big), run_time=self.rt(0.5))
        else:
            self.play(FadeIn(big, scale=1.05), run_time=self.rt(0.5))
        self.caption_mobj = big
        self.wait(self.rt(1.6))

    # ------------------------------------------------------------- summary
    def summary_row(self, op: str, n: int) -> str:
        return f"{op}  ×{n}   ({n // self.cfg.n_layers} per layer)"

    def dock_out_finale(self):
        """The finished Out decks land in the Out dock."""
        finals = self.acts.get("Out")
        if not finals:
            return
        self.play(
            *[
                g.animate.move_to(self.canvas.anchor("out", "dock", i))
                for i, g in enumerate(finals)
            ],
            run_time=0.7,
        )
        self.set_tracker(
            LTensor("Out", ("B", "T", "D"), self.cfg.act_sharding),
            self.canvas.anchor("out", "dock", 0)[0],
            run_time=0.3,
        )

    def summary(self):
        self.speed = 1.0
        self.dock_out_finale()
        steps = forward_steps(self.cfg)
        counts = count_collectives(steps)
        from manim import VMobject

        background = [
            m
            for m in self.mobjects
            if m is not self.strip and m is not self.highlight and isinstance(m, VMobject)
        ]
        fade_anims = [m.animate.set_opacity(0.14) for m in background]
        if self.highlight is not None:
            fade_anims.append(FadeOut(self.highlight))
            self.highlight = None
        self.play(*fade_anims, run_time=0.8)

        title = Text(
            "Forward-pass communication", font_size=32, color=style.TEXT_COLOR, weight="BOLD"
        )
        if counts:
            rows = [
                caption_text(self.summary_row(op, n), font_size=26, color=style.COMM_COLOR)
                for op, n in sorted(counts.items())
            ]
        else:
            rows = [caption_text("none — zero communication", font_size=28, color=style.GOOD_COLOR)]
        card = VGroup(title, *rows).arrange(DOWN, buff=0.35)
        card.move_to(UP * 0.3)
        card.set_z_index(style.Z_LABEL + 1)
        self.play(FadeIn(card, shift=UP * 0.2), run_time=0.8)
        self.wait(2.0)
