"""ForwardPassScene: plays a StrategyConfig's forward-pass step list.

Each concrete scene is a thin subclass setting `cfg`. Handlers are dispatched
by step class name (`handle_MatMulStep`, ...); pipeline-specific steps are
handled by the PP scene subclass.
"""

from __future__ import annotations

from typing import ClassVar

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    FadeIn,
    FadeOut,
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
from tpviz.mobjects.device_grid import DeviceGrid
from tpviz.mobjects.equation_strip import EquationStrip
from tpviz.mobjects.labels import caption_text, math_label, tensor_label
from tpviz.mobjects.tensor_mobject import ActivationDeck, WeightRect

PHASE_LABEL = {"attn": "Attention", "mlp": "MLP", "moe": "MoE MLP", "pipeline": "Pipeline"}
LAYER2_SPEED = 0.45


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
        self.weights: dict[str, list[VGroup]] = {}
        self.pending_restore: dict[str, list[VGroup]] = {}
        self.caption_mobj: Text | None = None
        self.badge: Text | None = None

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
                self.on_phase(step)
            self.speed = 1.0 if step.layer <= 1 else LAYER2_SPEED
            self.play_step(step)
        self.next_section("summary")
        self.summary()

    def rt(self, base: float) -> float:
        return base * self.speed

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
        header.to_corner(UP + LEFT, buff=0.25)
        header.set_z_index(style.Z_LABEL)
        self.play(ReplacementTransform(card, header), run_time=0.8)
        self.header = header

    def setup_stage(self):
        n = self.cfg.mesh.n_devices
        box_w = min(3.0, (12.9 - 0.35 * (n - 1)) / n)
        self.grid = DeviceGrid(self.cfg.mesh, box_width=box_w, box_height=4.4, buff=0.35)
        self.grid.shift(DOWN * 0.55)
        self.strip = EquationStrip()
        self.add(self.strip)

        self.comm_label = caption_text("collectives", font_size=16, color=style.MUTED_TEXT)
        self.comm_value = caption_text("0", font_size=30, color=style.GOOD_COLOR)
        counter = VGroup(self.comm_label, self.comm_value).arrange(DOWN, buff=0.08)
        counter.to_corner(UP + RIGHT, buff=0.25)
        self.add(counter)

        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in self.grid.boxes], lag_ratio=0.12),
            run_time=1.2,
        )

    def place_initial_tensors(self):
        cfg = self.cfg
        weight_names = ["W_qkv", "W_o", "W_in", "W_out"]

        # weights: identical layout on every device
        for i, box in enumerate(self.grid.boxes):
            cells = []
            for name in weight_names:
                t = make_weight(cfg, name)
                vis = WeightRect(t, cfg.mesh)
                label = vis.make_label(font_size=30)
                cells.append(VGroup(vis, label))
            grid = VGroup(*cells).arrange_in_grid(rows=2, cols=2, buff=(0.55, 0.35))
            box.fit_into(grid, "weights", attach=False)
            for name, cell in zip(weight_names, cells):
                self.weights.setdefault(name, []).append(cell)

        # the input batch: one full deck splits into per-device shards
        t_in = make_input(cfg)
        full = ActivationDeck(t_in, cfg.mesh, sharded=False)
        full_label = full.make_label(font_size=24)
        full_group = VGroup(full, full_label)
        full_group.move_to(UP * 2.2)

        shard_groups = []
        for box in self.grid.boxes:
            vis = ActivationDeck(t_in, cfg.mesh)
            label = vis.make_label(font_size=30)
            g = VGroup(vis, label)
            box.fit_into(g, "acts", attach=False)
            shard_groups.append(g)
        self.acts["In"] = shard_groups

        self.play(FadeIn(full_group, scale=1.1), run_time=self.rt(0.8))
        self.show_caption_text("Place the tensors: " + self.cfg.tagline)
        self.play(
            LaggedStart(
                *[
                    LaggedStart(
                        *[FadeIn(c, shift=DOWN * 0.15) for c in self.weights[n]], lag_ratio=0.05
                    )
                    for n in weight_names
                ],
                lag_ratio=0.25,
            ),
            run_time=self.rt(1.6),
        )
        self.play(
            *[TransformFromCopy(full, g) for g in shard_groups],
            FadeOut(full_group),
            run_time=self.rt(1.2),
        )
        self.wait(self.rt(0.4))

    # ------------------------------------------------------------- utilities
    def build_act_vis(self, t: LTensor, device: int, *, like: VGroup | None = None) -> VGroup:
        vis = ActivationDeck(t, self.cfg.mesh)
        label = vis.make_label(font_size=30)
        g = VGroup(vis, label)
        if like is not None:
            s = min(1.0, like.width / g.width, like.height / g.height)
            g.scale(max(s, 0.35))
            g.move_to(like.get_center())
        else:
            self.grid.box(device).fit_into(g, "acts", attach=False)
        return g

    def show_caption_text(self, text: str | None):
        if text is None or self.speed < 1.0:
            return
        cap = caption_text(text, font_size=23)
        if cap.width > 12.8:
            cap.scale(12.8 / cap.width)
        cap.move_to(UP * 2.55)
        if self.caption_mobj is None:
            self.play(FadeIn(cap), run_time=0.4)
        else:
            self.play(ReplacementTransform(self.caption_mobj, cap), run_time=0.4)
        self.caption_mobj = cap

    def on_phase(self, step: Step):
        label = f"Layer {step.layer} · {PHASE_LABEL.get(step.phase, step.phase)}"
        badge = Text(label, font_size=20, color=style.ACCENT, weight="BOLD")
        badge.next_to(self.header, DOWN, buff=0.15, aligned_edge=LEFT)
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
        self.play(*anims, run_time=self.rt(0.4))

    def comm_bump(self):
        self.comm_count += 1
        new = caption_text(str(self.comm_count), font_size=30, color=style.COMM_COLOR)
        new.move_to(self.comm_value.get_center())
        self.play(ReplacementTransform(self.comm_value, new), run_time=0.3)
        self.comm_value = new

    def play_step(self, step: Step):
        self.show_caption_text(step.caption or default_caption(step))
        getattr(self, f"handle_{type(step).__name__}")(step)

    # -------------------------------------------------------------- handlers
    def handle_MatMulStep(self, step: MatMulStep):
        weights = self.weights[step.b.name]
        acts = self.acts.pop(step.a.name)
        outs = [self.build_act_vis(step.out, i) for i in range(len(self.grid))]
        self.play(self.strip.show(step.tex()), run_time=self.rt(0.5))
        compute.animate_matmul(self, weights, acts, outs, run_time=self.rt(1.3))
        self.acts[step.out.name] = outs
        self.restore_after_use(step.b.name)

    def restore_after_use(self, weight_name: str):
        """FSDP: the gathered weight was temporary — discard it, shard returns."""
        originals = self.pending_restore.pop(weight_name, None)
        if originals is None:
            return
        gathered = self.weights[weight_name]
        self.play(
            *[FadeOut(g) for g in gathered],
            *[FadeIn(o) for o in originals],
            run_time=self.rt(0.7),
        )
        self.weights[weight_name] = originals

    def handle_SplitQKVStep(self, step: SplitQKVStep):
        src_vis = self.acts.pop(step.src.name)
        per_name: dict[str, list[VGroup]] = {"Q": [], "K": [], "V": []}
        anims = []
        for i, box in enumerate(self.grid.boxes):
            minis = []
            for t in (step.q, step.k, step.v):
                vis = ActivationDeck(t, self.cfg.mesh, scale=0.7)
                label = vis.make_label(font_size=34)
                minis.append(VGroup(vis, label))
            row = VGroup(*minis).arrange(RIGHT, buff=0.3)
            box.fit_into(row, "acts", attach=False)
            for name, m in zip(("Q", "K", "V"), minis):
                per_name[name].append(m)
            anims.append(ReplacementTransform(src_vis[i], row))
        self.play(
            self.strip.show(
                rf"{step.src.tex()} \,\to\, {step.q.tex()},\ {step.k.tex()},\ {step.v.tex()}"
            ),
            run_time=self.rt(0.5),
        )
        self.play(*anims, run_time=self.rt(0.9))
        self.acts.update(per_name)

    def handle_AttentionCoreStep(self, step: AttentionCoreStep):
        qkv_groups = [
            [self.acts["Q"][i], self.acts["K"][i], self.acts["V"][i]]
            for i in range(len(self.grid))
        ]
        for name in ("Q", "K", "V"):
            self.acts.pop(name)
        capsules, outs = [], []
        for i, box in enumerate(self.grid.boxes):
            c = compute.make_attn_capsule()
            c.move_to(box.slot_center("acts"))
            capsules.append(c)
            outs.append(self.build_act_vis(step.out, i))
        self.play(self.strip.show(step.tex()), run_time=self.rt(0.5))
        compute.animate_attention(self, qkv_groups, capsules, outs, run_time=self.rt(1.8))
        self.acts[step.out.name] = outs

    def handle_GeluStep(self, step: GeluStep):
        self.play(self.strip.show(step.tex()), run_time=self.rt(0.4))
        compute.animate_gelu(self, self.acts[step.src.name], run_time=self.rt(0.7))

    def handle_AllGatherStep(self, step: AllGatherStep):
        self.play(
            self.strip.show(step.tex(), color=style.COMM_COLOR), run_time=self.rt(0.5)
        )
        if step.src.kind == "weight":
            shards = self.weights[step.src.name]
            self.pending_restore[step.src.name] = [s.copy() for s in shards]
            results = []
            for i, s in enumerate(shards):
                vis = WeightRect(step.out, self.cfg.mesh)
                label = vis.make_label(font_size=30)
                g = VGroup(vis, label)
                sc = min(s.width / g.width, s.height / g.height)
                g.scale(sc).move_to(s.get_center())
                results.append(g)
            coll.animate_all_gather(self, shards, results, run_time=self.rt(1.5))
            self.weights[step.src.name] = results
        else:
            shards = self.acts.pop(step.src.name)
            results = [
                self.build_act_vis(step.out, i, like=shards[i]) for i in range(len(shards))
            ]
            coll.animate_all_gather(self, shards, results, run_time=self.rt(1.5))
            self.acts[step.out.name] = results
        self.comm_bump()

    def handle_ReduceScatterStep(self, step: ReduceScatterStep):
        self._exchange(step)

    def handle_AllReduceStep(self, step: AllReduceStep):
        self._exchange(step)

    def _exchange(self, step):
        self.play(self.strip.show(step.tex(), color=style.COMM_COLOR), run_time=self.rt(0.5))
        partials = self.acts.pop(step.src.name)
        results = [self.build_act_vis(step.out, i) for i in range(len(partials))]
        coll.animate_exchange_resolve(self, partials, results, run_time=self.rt(1.5))
        self.acts[step.out.name] = results
        self.comm_bump()

    def handle_AllToAllStep(self, step: AllToAllStep):
        self._exchange(step)

    def handle_RouteStep(self, step: RouteStep):
        self.play(self.strip.show(step.tex(), color=style.ACCENT), run_time=self.rt(0.4))
        decks = self.acts[step.tokens.name]
        self.play(
            *[
                Indicate(d, scale_factor=1.07, color=style.EXPERT_HUES[i % len(style.EXPERT_HUES)])
                for i, d in enumerate(decks)
            ],
            run_time=self.rt(0.9),
        )

    def handle_AnnotateStep(self, step: AnnotateStep):
        big = caption_text(step.text, font_size=30, color=style.ACCENT)
        if big.width > 12.5:
            big.scale(12.5 / big.width)
        big.move_to(UP * 2.55)
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

    def summary(self):
        steps = forward_steps(self.cfg)
        counts = count_collectives(steps)
        from manim import VMobject

        background = [
            m for m in self.mobjects if m is not self.strip and isinstance(m, VMobject)
        ]
        self.play(*[m.animate.set_opacity(0.14) for m in background], run_time=0.8)

        title = Text("Forward-pass communication", font_size=32, color=style.TEXT_COLOR, weight="BOLD")
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
