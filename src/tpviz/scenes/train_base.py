"""TrainScene: forward pass (compressed, saving activations) + backward pass.

What it adds over ForwardPassScene:
  - a stash row along each lane's bottom edge where saved activations park
    (the memory cost of training, consumed again by dW matmuls)
  - gradient tensors in rose flowing RIGHT -> LEFT through the same stations
  - weight-gradient rects appearing as rose "shadows" behind each weight
  - a matmul counter beside the collectives counter: backward = 2x forward
"""

from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    UP,
    FadeIn,
    FadeOut,
    Indicate,
    ReplacementTransform,
    Text,
    VGroup,
)

from tpviz import style
from tpviz.animations import compute
from tpviz.core.backward import count_matmuls, train_steps
from tpviz.core.model import count_collectives
from tpviz.core.steps import (
    AttentionBwdStep,
    GradInitStep,
    MatMulStep,
    MergeQKVStep,
    SaveActivationStep,
    Step,
)
from tpviz.core.tensors import LTensor
from tpviz.mobjects.canvas import FIXTURE_SCALE, MINI_SCALE
from tpviz.mobjects.labels import caption_text
from tpviz.mobjects.tensor_mobject import ActivationDeck, WeightRect
from tpviz.scenes.base import PHASE_LABEL, ForwardPassScene

# where a backward matmul's product lands (flow runs right -> left)
BWD_NEXT_ANCHOR = {
    ("attn", 0): "entry",  # dX leaves the station
    ("attn", 1): "core",  # dA feeds the attention-core backward
    ("mlp", 0): "entry",
    ("mlp", 1): "mid",  # dTmp pauses at the gelu
    ("moe", 0): "core",
    ("moe", 1): "core",
}

FWD_SPEED = 0.55
FWD_SPEED_L2 = 0.4


class TrainScene(ForwardPassScene):
    def steps(self) -> list[Step]:
        return train_steps(self.cfg)

    def speed_for(self, step: Step) -> float:
        if not step.backward:
            return FWD_SPEED if step.layer <= 1 else FWD_SPEED_L2
        # backward visits the last layer first — full pace there, brisker after
        return 1.0 if step.layer == self.cfg.n_layers else 0.6

    def captions_on(self) -> bool:
        return self._captions

    def construct(self):
        self.speed = 1.0
        self._captions = True
        self.comm_count = 0
        self.matmul_count = 0
        self.acts, self.weights, self.pending_restore = {}, {}, {}
        self.stash: dict[tuple[str, str], list[VGroup]] = {}  # (station, tensor name)
        self.caption_mobj = self.badge = self.tracker = self.highlight = None
        self.cur_station = None
        self._saved_note_shown = False

        self.next_section("title")
        self.title_card()
        self.next_section("setup")
        self.setup_stage()
        self.place_initial_tensors()
        cur = (None, None, None)
        for step in self.steps():
            marker = (step.layer, step.phase, step.backward)
            if marker != cur:
                cur = marker
                self.next_section(f"{'b' if step.backward else 'f'}{step.layer}-{step.phase}")
                self.speed = self.speed_for(step)
                self._captions = step.backward or isinstance(step, SaveActivationStep)
                self.on_phase(step)
            self.speed = self.speed_for(step)
            self._captions = step.backward
            self.play_step(step)
        self.next_section("summary")
        self.summary()

    # ------------------------------------------------------------- chrome
    def setup_stage(self):
        super().setup_stage()
        self.matmul_label = caption_text("matmuls", font_size=16, color=style.MUTED_TEXT)
        self.matmul_value = caption_text("0", font_size=30, color=style.TEXT_COLOR)
        counter = VGroup(self.matmul_label, self.matmul_value).arrange(DOWN, buff=0.08)
        counter.web_role = "chrome"
        counter.to_corner(UP + np.array([1.0, 0.0, 0.0]), buff=0.2)
        counter.shift(np.array([-1.5, 0.0, 0.0]))
        self.add(counter)

    def matmul_bump(self, backward: bool):
        self.matmul_count += 1
        color = style.GRAD_STROKE if backward else style.TEXT_COLOR
        new = caption_text(str(self.matmul_count), font_size=30, color=color)
        new.web_role = "chrome"
        new.move_to(self.matmul_value.get_center())
        self.play(ReplacementTransform(self.matmul_value, new), run_time=0.25)
        self.matmul_value = new

    def on_phase(self, step: Step):
        if not step.backward:
            super().on_phase(step)
            return
        key = self.canvas.station_key(step.layer, step.phase)
        self.cur_station = key
        label = f"◀ Backward · Layer {step.layer} · {PHASE_LABEL.get(step.phase, step.phase)}"
        badge = Text(label, font_size=20, color=style.GRAD_STROKE, weight="BOLD")
        badge.next_to(self.header, DOWN, buff=0.12, aligned_edge=np.array([-1.0, 0.0, 0.0]))
        badge.set_z_index(style.Z_LABEL)
        badge.web_role = "chrome"
        anims = []
        if self.caption_mobj is not None:
            anims.append(FadeOut(self.caption_mobj))
            self.caption_mobj = None
        anims.append(
            ReplacementTransform(self.badge, badge) if self.badge is not None else FadeIn(badge)
        )
        self.badge = badge
        hl = self.canvas.station_highlight(key)
        if self.highlight is None:
            self.highlight = hl
            self.add(hl)
            anims.append(FadeIn(hl))
        else:
            anims.append(self.highlight.animate.move_to(hl.get_center()))
        # gradients arrive from the RIGHT
        for vis_list in self.acts.values():
            for i, g in enumerate(vis_list):
                anims.append(g.animate.move_to(self.canvas.anchor(key, "exit", i)))
        self.play(*anims, run_time=self.rt(0.6))

    # ------------------------------------------------------------- helpers
    def build_tensor_vis(self, t: LTensor, lane: int, *, at=None, like=None) -> VGroup:
        if t.name.startswith("dW"):
            vis = WeightRect(t, self.cfg.mesh, device=lane, scale=FIXTURE_SCALE * 0.95)
            # name chip ON the rect: no free vertical space around the weight track
            chip = caption_text(t.name, font_size=11, color=style.TEXT_COLOR)
            if chip.width > vis.width * 1.15:
                chip.scale(vis.width * 1.15 / chip.width)
            chip.move_to(vis.get_center())
            chip.set_z_index(style.Z_LABEL)
            g = VGroup(vis, chip)
            if like is not None:
                s = min(1.0, like.width / max(g.width, 1e-6), like.height / max(g.height, 1e-6))
                g.scale(s)
                g.move_to(like.get_center())
            elif at is not None:
                g.move_to(at)
            return g
        return self.build_act_vis(t, lane, at=at, like=like)

    # ------------------------------------------------------------- handlers
    def handle_SaveActivationStep(self, step: SaveActivationStep):
        key = self.canvas.station_key(step.layer, step.phase)
        minis = []
        for i in range(self.n_lanes()):
            vis = ActivationDeck(step.t, self.cfg.mesh, device=i, scale=0.14)
            if vis.width > 0.24:
                vis.scale(0.24 / vis.width)
            vis.set_opacity(style.SAVED_OPACITY)
            vis.web_save = True
            chip = caption_text(step.t.name, font_size=11, color=style.MUTED_TEXT)
            chip.web_save = True
            g = VGroup(vis, chip.next_to(vis, DOWN, buff=0.03))
            g.move_to(self.canvas.stash_slot(key, i, step.anchor, step.spread))
            minis.append(g)
        self.stash[(key, step.t.name)] = minis
        if not self._saved_note_shown:
            self._saved_note_shown = True
            self._captions = True
            self.show_caption_text(
                "Every matmul input is SAVED for the backward pass — the memory cost of training"
            )
        self.play(*[FadeIn(m, scale=1.6) for m in minis], run_time=self.rt(0.35))

    def handle_GradInitStep(self, step: GradInitStep):
        finals = self.acts.pop("Out")
        douts = []
        for i in range(self.n_lanes()):
            g = self.build_act_vis(step.out, i, at=self.canvas.anchor("out", "dock", i))
            douts.append(g)
        self.play(self.strip.show(step.tex(), color=style.GRAD_STROKE), run_time=0.5)
        self.show_caption_text(step.caption)
        self.play(
            *[ReplacementTransform(f, d) for f, d in zip(finals, douts)], run_time=0.9
        )
        self.acts[step.out.name] = douts
        self.set_tracker(step.out, self.canvas.anchor("out", "dock", 0)[0])
        self.wait(0.4)

    def handle_MatMulStep(self, step: MatMulStep):
        if not step.backward:
            super().handle_MatMulStep(step)
            self.matmul_bump(False)
            return
        key = self.canvas.station_key(step.layer, step.phase)
        if step.b.kind == "grad":
            self._dw_matmul(step, key)
        else:
            self._dx_matmul(step, key)
        self.matmul_bump(True)

    def _dx_matmul(self, step: MatMulStep, key: str):
        """dX = dY · Wᵀ — the grad deck slides left under the (transposed) weight."""
        slot = 0 if step.b.name in ("W_qkv", "W_in") else 1
        fixtures = self.weights[(key, step.b.name)]
        acts = self.acts.pop(step.a.name)
        anchor_name = "core" if step.phase == "moe" else ("w1" if slot == 0 else "w2")
        self.play(
            *[
                g.animate.move_to(self.canvas.anchor(key, anchor_name, i))
                for i, g in enumerate(acts)
            ],
            self.strip.show(step.tex(), color=style.GRAD_STROKE, note=step.note_tex),
            run_time=self.rt(0.45),
        )
        out_anchor = BWD_NEXT_ANCHOR[(step.phase, slot)]
        outs = [
            self.build_tensor_vis(step.out, i, at=self.canvas.anchor(key, out_anchor, i))
            for i in range(len(acts))
        ]
        compute.animate_matmul(self, fixtures, acts, outs, run_time=self.rt(1.1))
        self.acts[step.out.name] = outs
        self.set_tracker(step.out, self.canvas.anchor(key, out_anchor, 0)[0])
        self.restore_after_use(key, step.b.name)

    def _dw_matmul(self, step: MatMulStep, key: str):
        """dW = Xᵀ · dY — the stashed activation pulses; a rose shadow appears
        behind the weight it is the gradient of."""
        w_name = step.out.name[1:]  # dW_out -> W_out
        slot = 0 if w_name in ("W_qkv", "W_in") else 1
        stash = self.stash.get((key, step.a.name), [])
        outs = [
            self.build_tensor_vis(step.out, i, at=self.canvas.grad_anchor(key, slot, i))
            for i in range(self.n_lanes())
        ]
        for o in outs:
            o.set_z_index(style.Z_TENSOR - 1)
        self.play(
            self.strip.show(step.tex(), color=style.GRAD_STROKE, note=step.note_tex),
            run_time=self.rt(0.4),
        )
        anims = [Indicate(m, scale_factor=1.4, color=style.ACT_STROKE) for m in stash]
        anims += [
            Indicate(w, scale_factor=1.06, color=style.WEIGHT_STROKE)
            for w in self.weights[(key, w_name)]
        ]
        anims += [FadeIn(o, shift=DOWN * 0.1) for o in outs]
        self.play(*anims, run_time=self.rt(1.0))
        self.acts[step.out.name] = outs

    def handle_AttentionBwdStep(self, step: AttentionBwdStep):
        key = self.canvas.station_key(step.layer, step.phase)
        da = self.acts.pop(step.da.name)
        per_name: dict[str, list[VGroup]] = {}
        anims = []
        for i in range(self.n_lanes()):
            minis = []
            for t in (step.dq, step.dk, step.dv):
                vis = ActivationDeck(t, self.cfg.mesh, device=i, scale=MINI_SCALE)
                chip = caption_text(t.name, font_size=13, color=style.GRAD_STROKE)
                chip.next_to(vis, DOWN, buff=0.05)
                minis.append(VGroup(vis, chip))
            row = VGroup(*minis).arrange(np.array([1.0, 0.0, 0.0]), buff=0.14, aligned_edge=UP)
            lane_h = self.canvas.lanes[i].rect.height
            s = min(1.0, 1.5 / max(row.width, 1e-6), lane_h * 0.50 / max(row.height, 1e-6))
            row.scale(s)
            row.move_to(self.canvas.anchor(key, "core", i))
            for t, m in zip((step.dq, step.dk, step.dv), minis):
                per_name.setdefault(t.name, []).append(m)
            anims.append(ReplacementTransform(da[i], row))
        # the saved Q, K, V pulse — the attention backward consumes them
        stash_pulses = []
        for name in ("Q", "K", "V"):
            for m in self.stash.get((key, name), []):
                stash_pulses.append(Indicate(m, scale_factor=1.4, color=style.KV_STROKE))
        self.play(
            self.strip.show(step.tex(), color=style.GRAD_STROKE, note=step.note_tex),
            run_time=self.rt(0.4),
        )
        self.play(*anims, *stash_pulses, run_time=self.rt(1.0))
        self.acts.update(per_name)

    def handle_MergeQKVStep(self, step: MergeQKVStep):
        key = self.canvas.station_key(step.layer, step.phase)
        parts = [self.acts.pop(t.name) for t in (step.q, step.k, step.v)]
        outs = [
            self.build_act_vis(step.out, i, at=self.canvas.anchor(key, "core", i))
            for i in range(self.n_lanes())
        ]
        self.play(self.strip.show(step.tex(), color=style.GRAD_STROKE), run_time=self.rt(0.35))
        self.play(
            *[
                ReplacementTransform(VGroup(parts[0][i], parts[1][i], parts[2][i]), outs[i])
                for i in range(self.n_lanes())
            ],
            run_time=self.rt(0.7),
        )
        self.acts[step.out.name] = outs
        self.set_tracker(step.out, self.canvas.anchor(key, "core", 0)[0])

    # ------------------------------------------------------------- summary
    def dock_out_finale(self):
        finals = self.acts.get("dX")
        if not finals:
            return
        din = LTensor("dIn", ("B", "T", "D"), self.cfg.act_sharding, kind="grad")
        outs = [
            self.build_act_vis(din, i, at=self.canvas.anchor("in", "dock", i))
            for i in range(self.n_lanes())
        ]
        self.play(
            *[ReplacementTransform(f, o) for f, o in zip(finals, outs)], run_time=0.7
        )
        self.set_tracker(din, self.canvas.anchor("in", "dock", 0)[0], run_time=0.3)

    def summary(self):
        self.speed = 1.0
        self._captions = True
        self.dock_out_finale()
        steps = self.steps()
        mm = count_matmuls(steps)
        fwd_coll = count_collectives([s for s in steps if not s.backward])
        bwd_coll = count_collectives([s for s in steps if s.backward])

        from manim import VMobject

        background = [
            m
            for m in self.mobjects
            if m is not self.strip and m is not self.highlight and isinstance(m, VMobject)
        ]
        fade = [m.animate.set_opacity(0.14) for m in background]
        if self.highlight is not None:
            fade.append(FadeOut(self.highlight))
            self.highlight = None
        self.play(*fade, run_time=0.8)

        def fmt(d: dict[str, int]) -> str:
            return "  ·  ".join(f"{k} ×{v}" for k, v in sorted(d.items())) if d else "none"

        title = Text("Training step: compute & communication", font_size=30,
                     color=style.TEXT_COLOR, weight="BOLD")
        rows = [
            caption_text(
                f"forward:   {mm['forward']} matmuls    collectives: {fmt(fwd_coll)}",
                font_size=24, color=style.ACT_STROKE,
            ),
            caption_text(
                f"backward: {mm['backward']} matmuls    collectives: {fmt(bwd_coll)}",
                font_size=24, color=style.GRAD_STROKE,
            ),
            caption_text(
                "backward ≈ 2× forward compute — plus the memory of every saved activation",
                font_size=22, color=style.MUTED_TEXT,
            ),
        ]
        card = VGroup(title, *rows).arrange(DOWN, buff=0.32)
        if card.width > 13.0:
            card.scale(13.0 / card.width)
        card.move_to(UP * 0.3)
        card.set_z_index(style.Z_LABEL + 1)
        self.play(FadeIn(card, shift=UP * 0.2), run_time=0.8)
        self.wait(2.2)
