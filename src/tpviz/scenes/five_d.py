"""FiveDScene: the full 5D combination on the MeshGrid canvas.

Every collective the engine derives is played as flights between the boxes of
exactly one mesh axis, wrapped in a halo of that axis's color, with an
on-canvas badge naming the axis and the parallelism it implements
("AllGather over X · FSDP"). That is the point of the figure: after the six
singles, this one shows which communication each strategy costs when they all
run at once.

Layout: stage 0 (layer 1) is the left 4 x 4 device grid, stage 1 (layer 2)
the right one; see mobjects/meshgrid.py for the row/column <-> axis mapping.
Activation saving is not drawn here (the singles cover the memory story).
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
    Square,
    SurroundingRectangle,
    Text,
    TransformFromCopy,
    VGroup,
)

from tpviz import configs, style
from tpviz.animations import compute
from tpviz.core.backward import count_matmuls, train_steps
from tpviz.core.mesh import StrategyConfig
from tpviz.core.model import count_collectives, make_input, make_weight, stage_of
from tpviz.core.steps import (
    AllGatherStep,
    AllReduceStep,
    AllToAllStep,
    AttentionBwdStep,
    AttentionCoreStep,
    CollectiveStep,
    GeluStep,
    GradInitStep,
    MatMulStep,
    MergeQKVStep,
    P2PSendStep,
    ReduceScatterStep,
    RouteStep,
    SaveActivationStep,
    SplitQKVStep,
    Step,
)
from tpviz.core.tensors import LTensor
from tpviz.mobjects.equation_strip import EquationStrip
from tpviz.mobjects.labels import caption_text, math_label
from tpviz.mobjects.meshgrid import MeshGrid
from tpviz.mobjects.tensor_mobject import ActivationDeck, WeightRect
from tpviz.scenes.base import PHASE_LABEL, PHASE_WEIGHTS, default_caption

DECK_SCALE = 0.40
FIXTURE_SCALE = 0.30
MINI_SCALE = 0.26
CAPTION_Y = 3.05
TOKENS_PER_DEVICE = 4  # 2 x 2 grid, token k -> expert k % 2
TOKEN_SIDE, TOKEN_PITCH = 0.16, 0.22

NEXT_ANCHOR = {(False, 0): "core", (False, 1): "exit", (True, 1): "core", (True, 0): "entry"}

AXIS_LONG = {
    "X": "FSDP", "Y": "tensor parallelism", "C": "context parallelism",
    "Z": "expert parallelism", "stage": "pipeline parallelism",
}


def axis_of_step(step: Step) -> str | None:
    if isinstance(step, P2PSendStep):
        return "stage"
    if isinstance(step, CollectiveStep):
        return step.axis
    return None


def attribution_caption(cfg: StrategyConfig, step: Step) -> str | None:
    """Captions that say WHICH parallelism a collective belongs to and why."""
    ax = axis_of_step(step)
    if ax is None:
        return None
    role = cfg.axis_roles.get(ax, ax)
    who = f"{AXIS_LONG.get(ax, role)} (axis {ax})"
    if isinstance(step, AllGatherStep):
        if step.src.kind == "weight":
            return (f"{who}: re-gather the weight for the backward matmul — it was discarded after the forward"
                    if step.backward else
                    f"{who}: gather this weight's shards just in time, use it, discard it")
        if step.src.kind == "kv":
            return (f"{who}: the saved K/V shards are re-gathered for the attention backward"
                    if step.backward else
                    f"{who}: every query needs every key and value — gather K/V across the sequence shards")
        if step.backward:
            return f"{who}: gather the incoming gradient over the tensor axis (dual of the forward ReduceScatter)"
        return f"{who}: gather the D-sharded activations before the matmul"
    if isinstance(step, ReduceScatterStep):
        if step.src.name.startswith("dW"):
            return f"{who}: the weight gradient scatters back onto the shards the weight lives in"
        if step.src.name in ("dK", "dV"):
            return f"{who}: every query shard contributed to every token's dK/dV — scatter them back over the sequence"
        if step.backward:
            return f"{who}: the partial dX resolves and re-shards on D (dual of the forward AllGather)"
        return f"{who}: the partial sums resolve — each device keeps its slice of D"
    if isinstance(step, AllReduceStep):
        if ax == "Z":
            return "Outside the MoE block the expert axis Z is plain data parallelism — attention weight gradients AllReduce over it"
        if ax == "C":
            return "The context axis C replicates every weight — its gradient contributions must be summed"
        return f"{who}: unreduced partial sums resolve everywhere"
    if isinstance(step, AllToAllStep):
        if step.direction == "dispatch":
            return (f"{who}: token gradients travel to the expert that processed them"
                    if step.backward else
                    f"{who}: each token travels to the device holding its expert — S stays sharded over X and C")
        return (f"{who}: token gradients return home" if step.backward
                else f"{who}: processed tokens return home to their batch/sequence positions")
    return step.caption


class FiveDScene(Scene):
    cfg: ClassVar[StrategyConfig] = configs.FIVE_D

    # ------------------------------------------------------------- lifecycle
    def construct(self):
        self.speed = 1.0
        self.comm_count = 0
        self.matmul_count = 0
        self.acts: dict[str, dict[int, VGroup]] = {}
        self.weights: dict[str, dict[int, VGroup]] = {}
        self.station_fixtures: dict[tuple[int, str], dict[str, dict[int, VGroup]]] = {}
        self.pending_restore: dict[str, dict[int, VGroup]] = {}
        self.caption_mobj = self.badge_mobj = self.highlight = None
        self.header_tex: dict[int, object] = {}
        self.cur_marker = None
        self.cur_station: tuple[int, str] | None = None
        self.active_stage = 0  # where the activations are (P2P moves it)
        self.shown_stage: int | None = None  # whose header is highlighted

        self.next_section("title")
        self.title_card()
        self.next_section("setup")
        self.setup_stage()
        self.place_initial_tensors()
        for step in train_steps(self.cfg):
            if isinstance(step, SaveActivationStep):
                continue
            if step.caption is None:
                step.caption = attribution_caption(self.cfg, step)
            marker = (step.layer, step.phase, step.backward)
            if marker != self.cur_marker:
                self.cur_marker = marker
                self.next_section(f"{'b' if step.backward else 'f'}{step.layer}-{step.phase}")
                self.speed = self.speed_for(step)
                self.on_phase(step)
            self.speed = self.speed_for(step)
            self.play_step(step)
        self.next_section("summary")
        self.summary()

    def speed_for(self, step: Step) -> float:
        if not step.backward:
            return 0.8 if step.layer <= 1 else 0.55
        return 0.8 if step.layer == self.cfg.n_layers else 0.55

    def rt(self, base: float) -> float:
        return base * self.speed

    def mark_step(self, step):  # recorder hooks (no-ops in renders)
        pass

    def flush_mark(self):
        pass

    def play_step(self, step: Step):
        self.show_caption_text(step.caption or default_caption(step))
        getattr(self, f"handle_{type(step).__name__}")(step)

    # ----------------------------------------------------------------- intro
    def title_card(self):
        name = Text(self.cfg.name, font_size=44, color=style.TEXT_COLOR, weight="BOLD")
        tag = Text(self.cfg.tagline, font_size=24, color=style.MUTED_TEXT)
        mesh = math_label(self.cfg.mesh.tex(), font_size=30, color=style.ACCENT)
        card = VGroup(name, tag, mesh).arrange(DOWN, buff=0.45)
        if card.width > 13:
            card.scale(13 / card.width)
        self.play(FadeIn(card, shift=UP * 0.3), run_time=1.0)
        self.wait(1.4)
        header = VGroup(
            Text(self.cfg.name, font_size=22, color=style.TEXT_COLOR, weight="BOLD"),
            math_label(self.cfg.mesh.tex(), font_size=20, color=style.ACCENT),
        ).arrange(RIGHT, buff=0.4)
        header.to_corner(UP + LEFT, buff=0.15)
        header.set_z_index(style.Z_LABEL)
        header.web_role = "chrome"
        self.play(ReplacementTransform(card, header), run_time=0.8)
        self.header = header

    def setup_stage(self):
        self.grid = MeshGrid(self.cfg)
        self.strip = EquationStrip()
        self.add(self.strip)

        # video-only chrome: collective counter + axis legend
        self.comm_label = caption_text("collectives", font_size=16, color=style.MUTED_TEXT)
        self.comm_value = caption_text("0", font_size=30, color=style.GOOD_COLOR)
        counter = VGroup(self.comm_label, self.comm_value).arrange(DOWN, buff=0.08)
        counter.to_corner(UP + RIGHT, buff=0.15)
        counter.web_role = "chrome"
        self.add(counter)
        legend = VGroup()
        for ax, role in self.cfg.axis_roles.items():
            t = Text(f"{ax} · {role}", font_size=14, color=style.AXIS_HUES[ax], weight="BOLD")
            legend.add(t)
        legend.arrange(RIGHT, buff=0.55)
        legend.move_to(np.array([0.0, self.grid.bottom - 0.52, 0.0]))
        legend.set_z_index(style.Z_LABEL)
        legend.web_role = "chrome"
        self.add(legend)

        self.play(FadeIn(self.grid), run_time=1.0)

    def place_initial_weights(self):
        """Every stage holds its layer's weights from the start — the first
        station's fixtures in every box, dimmed on the stages not yet active
        (on_phase adopts them as resident fixtures)."""
        anims = []
        for stage in range(self.grid.n_stages):
            layer = stage * self.cfg.n_layers // self.grid.n_stages + 1
            phase = "attn"
            fixtures: dict[str, dict[int, VGroup]] = {}
            for slot, name in enumerate(PHASE_WEIGHTS[phase]):
                t = make_weight(self.cfg, name)
                per_dev = {}
                for dev in self.grid.stage_devs(stage):
                    vis = VGroup(WeightRect(t, self.cfg.mesh, device=dev, scale=FIXTURE_SCALE))
                    self.grid.fit_weight(vis, dev, slot)
                    if stage != 0:
                        vis.set_opacity(0.35)
                    per_dev[dev] = vis
                    anims.append(FadeIn(vis))
                fixtures[name] = per_dev
            self.station_fixtures[(layer, phase)] = fixtures
        self.show_caption_text("Every stage holds its own layer's weights, each sharded over X (FSDP) and Y (TP)")
        self.play(LaggedStart(*anims, lag_ratio=0.01), run_time=1.2)

    def place_initial_tensors(self):
        self.place_initial_weights()
        t_in = make_input(self.cfg)
        full = ActivationDeck(t_in, self.cfg.mesh, sharded=False, scale=0.9)
        full_label = full.make_label(font_size=26)
        full_group = VGroup(full, full_label)
        full_group.move_to(np.array([self.grid.stage_center_x(0), 0.0, 0.0]))
        full_group.set_z_index(style.Z_FLYING)
        shards = {}
        for dev in self.grid.stage_devs(0):
            vis = ActivationDeck(t_in, self.cfg.mesh, device=dev, scale=DECK_SCALE)
            shards[dev] = self.grid.fit_act(VGroup(vis), dev, "entry")
        self.acts["In"] = shards
        self.show_caption_text("32 devices: FSDP × TP × CP × EP inside each stage, two pipeline stages")
        self.play(FadeIn(full_group, scale=1.1), run_time=0.7)
        self.wait(0.5)
        self.show_caption_text("The input is sharded 4 ways at once: batch over X and Z, sequence over C, width over Y")
        self.play(
            *[TransformFromCopy(full, g) for g in shards.values()],
            FadeOut(full_group),
            run_time=1.2,
        )

    # ------------------------------------------------------------- utilities
    def build_act_vis(self, t: LTensor, dev: int, *, at=None, like=None, scale=DECK_SCALE) -> VGroup:
        g = VGroup(ActivationDeck(t, self.cfg.mesh, device=dev, scale=scale))
        if like is not None:
            s = min(1.0, like.width / max(g.width, 1e-6), like.height / max(g.height, 1e-6))
            g.scale(max(s, 0.35))
            g.move_to(like.get_center())
        else:
            cap_w, cap_h = self.grid.box_w * 0.5, self.grid.box_h * 0.47
            g.scale(min(1.0, cap_w / max(g.width, 1e-6), cap_h / max(g.height, 1e-6)))
            if at is not None:
                g.move_to(at)
        return g

    def build_grad_weight(self, t: LTensor, dev: int, *, at=None, like=None) -> VGroup:
        vis = WeightRect(t, self.cfg.mesh, device=dev, scale=FIXTURE_SCALE * 0.95)
        g = VGroup(vis)
        if like is not None:
            s = min(1.0, like.width / max(g.width, 1e-6), like.height / max(g.height, 1e-6))
            g.scale(s)
            g.move_to(like.get_center())
        else:
            self.grid.fit_weight(g, dev, 0)
            if at is not None:
                g.move_to(at)
        return g

    def captions_on(self) -> bool:
        return True

    def show_caption_text(self, text: str | None):
        if text is None:
            return
        cap = caption_text(text, font_size=21)
        cap.web_role = "chrome"
        if cap.width > 13.4:
            cap.scale(13.4 / cap.width)
        cap.move_to(UP * CAPTION_Y)
        if self.caption_mobj is None:
            self.play(FadeIn(cap), run_time=0.35)
        else:
            self.play(ReplacementTransform(self.caption_mobj, cap), run_time=0.35)
        self.caption_mobj = cap

    def comm_bump(self):
        self.comm_count += 1
        new = caption_text(str(self.comm_count), font_size=30, color=style.COMM_COLOR)
        new.web_role = "chrome"
        new.move_to(self.comm_value.get_center())
        self.play(ReplacementTransform(self.comm_value, new), run_time=0.25)
        self.comm_value = new

    def badge_anim(self, step: Step | None):
        """The on-canvas attribution badge under the active grid — the axis
        and the parallelism responsible for the current collective."""
        ax = axis_of_step(step) if step is not None else None
        if ax is None:
            if self.badge_mobj is None:
                return None
            old, self.badge_mobj = self.badge_mobj, None
            return FadeOut(old)
        role = self.cfg.axis_roles.get(ax, ax)
        if isinstance(step, P2PSendStep):
            text = f"P2P send · stage {step.src_stage} → {step.dst_stage} · {role}"
        else:
            text = f"{step.op} over {ax}  ·  {role}"
        badge = Text(text, font_size=16, color=style.AXIS_HUES[ax], weight="BOLD")
        badge.move_to(self.grid.badge_pos(self.active_stage))
        badge.set_z_index(style.Z_LABEL)
        if self.badge_mobj is None:
            self.badge_mobj = badge
            return FadeIn(badge)
        old, self.badge_mobj = self.badge_mobj, badge
        return ReplacementTransform(old, badge)

    def op_intro(self, step: Step, *, color: str, note: str | None = None, run_time: float = 0.4):
        """Show the op in the strip and update the badge in one beat."""
        anims = [self.strip.show(step.tex(), color=color, note=note)]
        b = self.badge_anim(step)
        if b is not None:
            anims.append(b)
        self.play(*anims, run_time=self.rt(run_time))

    def station_tex(self, phase: str, override: LTensor | None = None) -> str:
        parts = []
        for name in PHASE_WEIGHTS[phase]:
            t = make_weight(self.cfg, name)
            if override is not None and override.name == name:
                t = override
            parts.append(t.tex())
        return r"\quad ".join(parts)

    def set_header_tex(self, stage: int, tex: str):
        """Returns an animation swapping the stage's weight-notation line."""
        m = math_label(tex, font_size=20, color=style.MUTED_TEXT)
        max_w = self.grid.grid_w - 0.2
        if m.width > max_w:
            m.scale(max_w / m.width)
        m.move_to(np.array([self.grid.stage_center_x(stage), self.grid.header_y(), 0.0]))
        old = self.header_tex.get(stage)
        self.header_tex[stage] = m
        if old is None:
            return FadeIn(m)
        return ReplacementTransform(old, m)

    def set_stage_title(self, stage: int, text: str, color: str):
        t = caption_text(text, font_size=18, color=color)
        t.move_to(self.grid.stage_titles[stage].get_center())
        old = self.grid.stage_titles[stage]
        self.grid.stage_titles[stage] = t
        return ReplacementTransform(old, t)

    def devs(self) -> list[int]:
        return self.grid.stage_devs(self.active_stage)

    # -------------------------------------------------------------- phases
    def on_phase(self, step: Step):
        layer, phase, bwd = step.layer, step.phase, step.backward
        anims = []
        if self.caption_mobj is not None:
            anims.append(FadeOut(self.caption_mobj))
            self.caption_mobj = None
        b = self.badge_anim(None)
        if b is not None:
            anims.append(b)
        if phase == "pipeline":
            self.play(*anims, run_time=self.rt(0.3))
            return

        stage = stage_of(self.cfg, layer)
        prev_stage = self.shown_stage
        self.active_stage = self.shown_stage = stage
        title = f"Stage {stage} · Layer {layer} · {PHASE_LABEL.get(phase, phase)}"
        if bwd:
            title = "◀ Backward · " + title
        anims.append(self.set_stage_title(stage, title, style.GRAD_STROKE if bwd else style.ACCENT))
        if prev_stage is not None and prev_stage != stage:
            anims.append(self.set_stage_title(prev_stage, f"Stage {prev_stage} · Layer {prev_stage + 1}",
                                              style.MUTED_TEXT))
            if prev_stage in self.header_tex:
                anims.append(FadeOut(self.header_tex.pop(prev_stage)))

        hl = self.grid.stage_highlight(stage)
        if self.highlight is None:
            self.highlight = hl
            self.add(hl)
            anims.append(FadeIn(hl))
        else:
            anims.append(self.highlight.animate.move_to(hl.get_center()))

        station = (layer, phase)
        if station != self.cur_station:
            # leave the previous station: same stage -> its weights go away;
            # other stage -> they stay resident, dimmed (the stage still owns its layer)
            if self.cur_station is not None:
                old = {name: fx for name, fx in self.weights.items()}
                old.update({n: g for n, g in self.acts.items() if n.startswith("dW")})
                for n in [n for n in self.acts if n.startswith("dW")]:
                    del self.acts[n]
                self.weights = {}
                if self.cur_station[0] == layer or stage_of(self.cfg, self.cur_station[0]) == stage:
                    anims += [FadeOut(g) for fx in old.values() for g in fx.values()]
                else:
                    self.station_fixtures[self.cur_station] = old
                    anims += [g.animate.set_opacity(0.35) for fx in old.values() for g in fx.values()]
            self.cur_station = station
            resident = self.station_fixtures.pop(station, None)
            if resident is not None:
                for name, fx in resident.items():
                    if name.startswith("dW"):
                        self.acts[name] = fx
                    else:
                        self.weights[name] = fx
                    anims += [g.animate.set_opacity(1.0) for g in fx.values()]
            else:
                for slot, name in enumerate(PHASE_WEIGHTS[phase]):
                    t = make_weight(self.cfg, name)
                    per_dev = {}
                    for dev in self.grid.stage_devs(stage):
                        vis = VGroup(WeightRect(t, self.cfg.mesh, device=dev, scale=FIXTURE_SCALE))
                        self.grid.fit_weight(vis, dev, slot)
                        per_dev[dev] = vis
                        anims.append(FadeIn(vis))
                    self.weights[name] = per_dev
            anims.append(self.set_header_tex(stage, self.station_tex(phase)))
        # live activations settle at the station's entry (forward) / exit (backward)
        for name, per_dev in self.acts.items():
            if name.startswith("dW"):
                continue
            for dev, g in per_dev.items():
                anims.append(g.animate.move_to(self.grid.anchor(dev, "exit" if bwd else "entry")))
        self.play(*anims, run_time=self.rt(0.7))

    # ----------------------------------------------------------- flights
    def flight(self, src: VGroup, dest, axis: str, *, arc: float = 0.5, scale: float = 0.85,
               opacity: float | None = None) -> tuple[VGroup, object]:
        """A copy of `src`'s body flies to `dest` inside a halo colored by the
        mesh axis the collective runs over."""
        body = src[0] if isinstance(src, VGroup) and len(src) else src
        c = body.copy()
        if opacity is not None:
            c.set_opacity(opacity)
        c.scale(scale)
        halo = SurroundingRectangle(c, color=style.AXIS_HUES[axis], buff=0.035, stroke_width=2.4)
        halo.set_fill(opacity=0.0)
        g = VGroup(c, halo)
        g.set_z_index(style.Z_FLYING)
        halo.set_z_index(style.Z_FLYING + 1)
        return g, g.animate(path_arc=arc).move_to(dest)

    def pulse_axis_labels(self, axis: str):
        return [Indicate(t, scale_factor=1.35, color=style.AXIS_HUES[axis])
                for t in self.grid.margin_labels.get(axis, [])]

    def all_gather(self, axis: str, shards: dict[int, VGroup], results: dict[int, VGroup],
                   *, run_time: float):
        copies, flights = [], []
        for j, res in results.items():
            for i in self.grid.group(j, axis):
                if i == j or i not in shards:
                    continue
                c, a = self.flight(shards[i], res.get_center(), axis)
                copies.append(c)
                flights.append(a)
        self.add(*copies)
        self.play(*flights, *self.pulse_axis_labels(axis), run_time=run_time * 0.65)
        self.play(
            FadeOut(VGroup(*copies)),
            *[ReplacementTransform(shards[j], results[j]) for j in results],
            run_time=run_time * 0.35,
        )

    def exchange(self, axis: str, partials: dict[int, VGroup], results: dict[int, VGroup],
                 *, run_time: float):
        copies, flights = [], []
        for j, res in results.items():
            for i in self.grid.group(j, axis):
                if i == j or i not in partials:
                    continue
                c, a = self.flight(partials[i], res.get_center(), axis, arc=0.35, scale=0.45, opacity=0.35)
                copies.append(c)
                flights.append(a)
        self.add(*copies)
        self.play(*flights, *self.pulse_axis_labels(axis), run_time=run_time * 0.6)
        self.play(
            FadeOut(VGroup(*copies)),
            *[ReplacementTransform(partials[j], results[j]) for j in results],
            run_time=run_time * 0.4,
        )

    # -------------------------------------------------------------- handlers
    def handle_MatMulStep(self, step: MatMulStep):
        self.matmul_count += 1
        if step.b.kind == "grad":
            self._dw_matmul(step)
            return
        slot = 0 if step.b.name in ("W_qkv", "W_in") else 1
        fixtures = self.weights[step.b.name]
        acts = self.acts.pop(step.a.name)
        color = style.GRAD_STROKE if step.backward else style.TEXT_COLOR
        self.play(
            *[g.animate.move_to(self.grid.weight_anchor(dev, slot) + DOWN * (self.grid.box_h * 0.42))
              for dev, g in acts.items()],
            self.strip.show(step.tex(), color=color, note=step.note_tex),
            *([self.badge_anim(None)] if self.badge_mobj is not None else []),
            run_time=self.rt(0.45),
        )
        out_anchor = NEXT_ANCHOR[(step.backward, slot)]
        outs = {dev: self.build_act_vis(step.out, dev, at=self.grid.anchor(dev, out_anchor))
                for dev in acts}
        compute.animate_matmul(
            self, [fixtures[d] for d in acts], list(acts.values()), list(outs.values()),
            run_time=self.rt(1.1),
        )
        self.acts[step.out.name] = outs
        self.restore_after_use(step.b.name)

    def _dw_matmul(self, step: MatMulStep):
        w_name = step.out.name[1:]
        slot = 0 if w_name in ("W_qkv", "W_in") else 1
        outs = {}
        for dev in self.devs():
            g = self.build_grad_weight(step.out, dev, at=self.grid.grad_anchor(dev, slot))
            g.set_z_index(style.Z_TENSOR - 1)
            outs[dev] = g
        self.play(
            self.strip.show(step.tex(), color=style.GRAD_STROKE, note=step.note_tex),
            *([self.badge_anim(None)] if self.badge_mobj is not None else []),
            run_time=self.rt(0.4),
        )
        anims = [Indicate(w, scale_factor=1.06, color=style.WEIGHT_STROKE)
                 for w in self.weights[w_name].values()]
        anims += [FadeIn(o, shift=DOWN * 0.08) for o in outs.values()]
        self.play(*anims, run_time=self.rt(0.9))
        self.acts[step.out.name] = outs

    def restore_after_use(self, weight_name: str):
        originals = self.pending_restore.pop(weight_name, None)
        if originals is None:
            return
        gathered = self.weights[weight_name]
        self.play(
            *[FadeOut(g) for g in gathered.values()],
            *[FadeIn(o) for o in originals.values()],
            self.set_header_tex(self.active_stage, self.station_tex(self.cur_station[1])),
            run_time=self.rt(0.5),
        )
        self.weights[weight_name] = originals

    def _minis_row(self, tensors, dev: int, at, color: str) -> list[VGroup]:
        # no name chips here (unlike the singles): a Text inside a morph target
        # makes Manim align hundreds of glyph submobjects per device
        minis = [VGroup(ActivationDeck(t, self.cfg.mesh, device=dev, scale=MINI_SCALE))
                 for t in tensors]
        row = VGroup(*minis).arrange(RIGHT, buff=0.1, aligned_edge=UP)
        s = min(1.0, self.grid.box_w * 0.8 / max(row.width, 1e-6),
                self.grid.box_h * 0.5 / max(row.height, 1e-6))
        row.scale(s)
        row.move_to(at)
        return minis

    def handle_SplitQKVStep(self, step: SplitQKVStep):
        src = self.acts.pop(step.src.name)
        per_name: dict[str, dict[int, VGroup]] = {"Q": {}, "K": {}, "V": {}}
        anims = []
        for dev, g in src.items():
            minis = self._minis_row((step.q, step.k, step.v), dev, self.grid.anchor(dev, "core"),
                                    style.KV_STROKE)
            for name, m in zip(("Q", "K", "V"), minis):
                per_name[name][dev] = m
            anims.append(FadeTransform(g, VGroup(*minis)))
        self.op_intro(step, color=style.TEXT_COLOR)
        self.play(*anims, run_time=self.rt(0.7))
        self.acts.update(per_name)

    def handle_AttentionCoreStep(self, step: AttentionCoreStep):
        groups, capsules, outs = {}, {}, {}
        for dev in list(self.acts["Q"]):
            groups[dev] = [self.acts["Q"][dev], self.acts["K"][dev], self.acts["V"][dev]]
            c = compute.make_attn_capsule()
            c.scale(min(1.0, self.grid.box_h * 0.42 / c.height, self.grid.box_w * 0.5 / c.width))
            c.move_to(self.grid.anchor(dev, "core"))
            capsules[dev] = c
            outs[dev] = self.build_act_vis(step.out, dev, at=self.grid.anchor(dev, "core"))
        for name in ("Q", "K", "V"):
            self.acts.pop(name)
        self.op_intro(step, color=style.TEXT_COLOR)
        rt = self.rt(1.4)
        self.play(
            *[FadeIn(c, scale=0.8) for c in capsules.values()],
            *[m.animate.scale(0.1).move_to(capsules[dev].get_center()).set_opacity(0.0)
              for dev, grp in groups.items() for m in grp],
            run_time=rt * 0.55,
        )
        for grp in groups.values():
            for m in grp:
                self.remove(m)
        self.play(*[FadeTransform(capsules[d], outs[d]) for d in outs], run_time=rt * 0.45)
        self.acts[step.out.name] = outs

    def handle_GeluStep(self, step: GeluStep):
        color = style.GRAD_STROKE if step.backward else style.TEXT_COLOR
        self.op_intro(step, color=color, note=step.note_tex, run_time=0.35)
        vis = self.acts.pop(step.src.name)
        compute.animate_gelu(self, list(vis.values()), run_time=self.rt(0.55))
        self.acts[step.out.name] = vis

    def handle_AllGatherStep(self, step: AllGatherStep):
        self.op_intro(step, color=style.COMM_COLOR, note=step.note_tex)
        if step.src.kind == "weight":
            slot = 0 if step.src.name in ("W_qkv", "W_in") else 1
            shards = self.weights[step.src.name]
            self.pending_restore[step.src.name] = {d: s.copy() for d, s in shards.items()}
            results = {}
            for dev in shards:
                vis = VGroup(WeightRect(step.out, self.cfg.mesh, device=dev, scale=FIXTURE_SCALE))
                self.grid.fit_weight(vis, dev, slot)
                results[dev] = vis
            self.play(self.set_header_tex(self.active_stage,
                                          self.station_tex(self.cur_station[1], override=step.out)),
                      run_time=self.rt(0.3))
            self.all_gather(step.axis, shards, results, run_time=self.rt(1.3))
            self.weights[step.src.name] = results
        elif step.backward and step.src.kind == "kv":
            # the saved K/V shards return beside the core and re-gather
            dx = -0.22 if step.src.name == "K" else 0.22
            shards, results = {}, {}
            for dev in self.devs():
                at = self.grid.anchor(dev, "core") + np.array([dx, 0.0, 0.0])
                shards[dev] = self.build_act_vis(step.src, dev, at=at, scale=MINI_SCALE)
                results[dev] = self.build_act_vis(step.out, dev, at=at, scale=MINI_SCALE)
            self.play(*[FadeIn(s, scale=1.4) for s in shards.values()], run_time=self.rt(0.35))
            self.all_gather(step.axis, shards, results, run_time=self.rt(1.2))
            self.acts[step.out.name] = results
        else:
            shards = self.acts.pop(step.src.name)
            results = {dev: self.build_act_vis(step.out, dev, like=g) for dev, g in shards.items()}
            self.all_gather(step.axis, shards, results, run_time=self.rt(1.3))
            self.acts[step.out.name] = results
        self.comm_bump()

    def _exchange(self, step):
        self.op_intro(step, color=style.COMM_COLOR, note=step.note_tex)
        partials = self.acts.pop(step.src.name)
        if step.src.name.startswith("dW"):
            results = {dev: self.build_grad_weight(step.out, dev, like=g) for dev, g in partials.items()}
            for r in results.values():
                r.set_z_index(style.Z_TENSOR - 1)
        else:
            results = {dev: self.build_act_vis(step.out, dev, like=g) for dev, g in partials.items()}
        self.exchange(step.axis, partials, results, run_time=self.rt(1.3))
        self.acts[step.out.name] = results
        self.comm_bump()

    def handle_ReduceScatterStep(self, step: ReduceScatterStep):
        self._exchange(step)

    def handle_AllReduceStep(self, step: AllReduceStep):
        self._exchange(step)

    def handle_RouteStep(self, step: RouteStep):
        self.op_intro(step, color=style.ACCENT, run_time=0.35)
        decks = self.acts[step.tokens.name]
        self.play(*[Indicate(d, scale_factor=1.07, color=style.ACCENT) for d in decks.values()],
                  run_time=self.rt(0.6))

    # ---------------------------------------------------------------- tokens
    def _token_points(self, dev: int) -> list[np.ndarray]:
        c = self.grid.anchor(dev, "core")
        pts = []
        for r in range(2):
            for k in range(2):
                pts.append(c + np.array([(k - 0.5) * TOKEN_PITCH, (0.5 - r) * TOKEN_PITCH, 0.0]))
        return pts

    def _token(self, expert: int, *, grad: bool) -> Square:
        sq = Square(side_length=TOKEN_SIDE, stroke_width=1.6 if grad else 1.1)
        sq.set_fill(style.EXPERT2_HUES[expert % 2], opacity=0.7 if grad else 0.95)
        sq.set_stroke(style.GRAD_STROKE if grad else style.TEXT_COLOR, opacity=0.9 if grad else 0.5)
        sq.set_z_index(style.Z_FLYING)
        return sq

    def handle_AllToAllStep(self, step: AllToAllStep):
        self.op_intro(step, color=style.COMM_COLOR, note=step.note_tex)
        axis, grad = step.axis, step.backward
        n_exp = self.cfg.moe.n_experts
        src = self.acts.pop(step.src.name)
        if step.direction == "dispatch":
            # each deck explodes into tokens colored by their expert; the ones
            # whose expert lives on the partner device fly over (along Z)
            tokens: dict[int, list[tuple[Square, int]]] = {}
            anims = []
            for dev, g in src.items():
                pts = self._token_points(dev)
                toks = [(self._token(k % n_exp, grad=grad).move_to(pts[k]), k % n_exp)
                        for k in range(TOKENS_PER_DEVICE)]
                tokens[dev] = toks
                anims.append(FadeTransform(g, VGroup(*[t for t, _ in toks])))
            self.play(*anims, run_time=self.rt(0.6))
            arrivals: dict[int, list[Square]] = {d: [] for d in src}
            flights = []
            for dev, toks in tokens.items():
                grp = self.grid.group(dev, axis)
                for tok, e in toks:
                    dest = grp[e % len(grp)]
                    slot = len(arrivals[dest])
                    arrivals[dest].append(tok)
                    flights.append(tok.animate(path_arc=0.35 if dest != dev else 0.0)
                                   .move_to(self._token_points(dest)[slot % TOKENS_PER_DEVICE]))
            self.play(*flights, *self.pulse_axis_labels(axis), run_time=self.rt(1.4))
            outs = {dev: self.build_act_vis(step.out, dev, at=self.grid.anchor(dev, "core"))
                    for dev in src}
            self.play(*[FadeTransform(VGroup(*arrivals[d]), outs[d]) for d in src],
                      run_time=self.rt(0.6))
        else:
            # the expert's batch explodes (all tokens are this expert's color)
            # and every token flies home to its origin along Z
            land = "entry" if grad else "exit"
            home: dict[int, list[tuple[Square, int]]] = {d: [] for d in src}
            anims = []
            for dev, g in src.items():
                e = self.grid.mesh.coord(axis, dev)
                pts = self._token_points(dev)
                grp = self.grid.group(dev, axis)
                toks = []
                k = 0
                for origin in grp:
                    for _ in range(TOKENS_PER_DEVICE // len(grp)):
                        tok = self._token(e, grad=grad).move_to(pts[k])
                        toks.append(tok)
                        home[origin].append((tok, e))
                        k += 1
                anims.append(FadeTransform(g, VGroup(*toks)))
            self.play(*anims, run_time=self.rt(0.6))
            flights = []
            for origin, toks in home.items():
                pts = self._token_points(origin)
                per_expert = [0] * n_exp
                for tok, e in toks:
                    k = e + n_exp * per_expert[e]  # back to the slot it left from
                    per_expert[e] += 1
                    flights.append(tok.animate(path_arc=-0.35).move_to(pts[k % TOKENS_PER_DEVICE]))
            self.play(*flights, *self.pulse_axis_labels(axis), run_time=self.rt(1.4))
            outs = {dev: self.build_act_vis(step.out, dev, at=self.grid.anchor(dev, land))
                    for dev in src}
            self.play(*[FadeTransform(VGroup(*[t for t, _ in home[d]]), outs[d]) for d in src],
                      run_time=self.rt(0.6))
        self.acts[step.out.name] = outs
        self.comm_bump()

    # -------------------------------------------------------------- pipeline
    def handle_P2PSendStep(self, step: P2PSendStep):
        self.op_intro(step, color=style.COMM_COLOR, note=step.note_tex)
        decks = self.acts.pop(step.tensor.name)
        moved, anims, halos = {}, [], []
        land = "exit" if step.backward else "entry"
        for dev, g in decks.items():
            dst = self.grid.peer_on_stage(dev, step.dst_stage)
            target = self.grid.anchor(dst, land)
            halo = SurroundingRectangle(g, color=style.AXIS_HUES["stage"], buff=0.03, stroke_width=2.2)
            halo.set_fill(opacity=0.0)
            halo.set_z_index(style.Z_FLYING + 1)
            halos.append(halo)
            arc = 0.22 if step.backward else -0.22
            anims.append(g.animate(path_arc=arc).move_to(target))
            anims.append(halo.animate(path_arc=arc).move_to(target))
            moved[dst] = g
        self.add(*halos)
        hl = self.grid.stage_highlight(step.dst_stage)
        self.active_stage = step.dst_stage
        self.play(*anims, self.highlight.animate.move_to(hl.get_center()),
                  *self.pulse_axis_labels("stage"), run_time=self.rt(1.3))
        self.play(FadeOut(VGroup(*halos)), run_time=self.rt(0.3))
        self.acts[step.tensor.name] = moved
        self.comm_bump()

    # -------------------------------------------------------------- backward
    def handle_GradInitStep(self, step: GradInitStep):
        finals = self.acts.pop("Out")
        douts = {dev: self.build_act_vis(step.out, dev, like=g) for dev, g in finals.items()}
        self.op_intro(step, color=style.GRAD_STROKE, run_time=0.5)
        self.play(*[ReplacementTransform(finals[d], douts[d]) for d in finals], run_time=self.rt(0.9))
        self.acts[step.out.name] = douts
        self.wait(0.3)

    def handle_AttentionBwdStep(self, step: AttentionBwdStep):
        da = self.acts.pop(step.da.name)
        per_name: dict[str, dict[int, VGroup]] = {}
        anims = []
        for dev, g in da.items():
            minis = self._minis_row((step.dq, step.dk, step.dv), dev, self.grid.anchor(dev, "core"),
                                    style.GRAD_STROKE)
            for t, m in zip((step.dq, step.dk, step.dv), minis):
                per_name.setdefault(t.name, {})[dev] = m
            anims.append(FadeTransform(g, VGroup(*minis)))
        for name in ("K", "V"):  # the re-gathered full K/V are consumed here
            if name in self.acts:
                anims += [FadeOut(g) for g in self.acts.pop(name).values()]
        self.op_intro(step, color=style.GRAD_STROKE, note=step.note_tex)
        self.play(*anims, run_time=self.rt(0.9))
        self.acts.update(per_name)

    def handle_MergeQKVStep(self, step: MergeQKVStep):
        parts = [self.acts.pop(t.name) for t in (step.q, step.k, step.v)]
        outs = {dev: self.build_act_vis(step.out, dev, at=self.grid.anchor(dev, "core"))
                for dev in parts[0]}
        self.op_intro(step, color=style.GRAD_STROKE, note=step.note_tex, run_time=0.35)
        self.play(*[FadeTransform(VGroup(parts[0][d], parts[1][d], parts[2][d]), outs[d])
                    for d in outs], run_time=self.rt(0.6))
        self.acts[step.out.name] = outs

    # --------------------------------------------------------------- summary
    def summary(self):
        self.speed = 1.0
        steps = train_steps(self.cfg)
        per_axis: dict[str, dict[str, int]] = {}
        for s in steps:
            ax = axis_of_step(s)
            if ax is None:
                continue
            op = "P2P" if isinstance(s, P2PSendStep) else s.op
            per_axis.setdefault(ax, {})[op] = per_axis.setdefault(ax, {}).get(op, 0) + 1
        from manim import VMobject

        background = [m for m in self.mobjects
                      if m is not self.strip and m is not self.highlight and isinstance(m, VMobject)]
        fade = [m.animate.set_opacity(0.12) for m in background]
        if self.highlight is not None:
            fade.append(FadeOut(self.highlight))
            self.highlight = None
        self.play(*fade, run_time=0.8)
        title = Text("One training step, every collective attributed", font_size=30,
                     color=style.TEXT_COLOR, weight="BOLD")
        rows = [title]
        for ax, role in self.cfg.axis_roles.items():
            ops = per_axis.get(ax, {})
            desc = "  ·  ".join(f"{op} ×{n}" for op, n in sorted(ops.items())) or "none"
            rows.append(caption_text(f"{ax} · {role}:   {desc}", font_size=22, color=style.AXIS_HUES[ax]))
        mm = count_matmuls(steps)
        rows.append(caption_text(
            f"{mm['forward']} forward + {mm['backward']} backward matmuls; "
            f"{sum(count_collectives(steps).values())} collectives in all",
            font_size=20, color=style.MUTED_TEXT))
        card = VGroup(*rows).arrange(DOWN, buff=0.3)
        if card.width > 13.0:
            card.scale(13.0 / card.width)
        card.move_to(UP * 0.2)
        card.set_z_index(style.Z_LABEL + 1)
        self.play(FadeIn(card, shift=UP * 0.2), run_time=0.8)
        self.wait(2.5)


class FiveDDebug(Scene):
    """Static geometry sign-off: the grid with fixtures + decks in every box."""

    cfg = configs.FIVE_D

    def construct(self):
        grid = MeshGrid(self.cfg)
        self.add(grid)
        t_in = make_input(self.cfg)
        for dev in grid.boxes:
            phase = "attn" if grid.stage_of(dev) == 0 else "moe"
            for slot, name in enumerate(PHASE_WEIGHTS[phase]):
                vis = VGroup(WeightRect(make_weight(self.cfg, name), self.cfg.mesh, device=dev,
                                        scale=FIXTURE_SCALE))
                self.add(grid.fit_weight(vis, dev, slot))
            deck = VGroup(ActivationDeck(t_in, self.cfg.mesh, device=dev, scale=DECK_SCALE))
            self.add(grid.fit_act(deck, dev, "core"))
        for s in range(2):
            m = math_label(r"W_{\mathrm{qkv}}[D_{X},\, H_{Y}] \quad W_{\mathrm{o}}[H_{Y},\, D_{X}]",
                           font_size=20, color=style.MUTED_TEXT)
            m.move_to(np.array([grid.stage_center_x(s), grid.header_y(), 0.0]))
            self.add(m)
        badge = Text("AllGather over X  ·  FSDP", font_size=16, color=style.AXIS_HUES["X"], weight="BOLD")
        badge.move_to(grid.badge_pos(0))
        self.add(badge, grid.stage_highlight(0))
        self.add(EquationStrip())
