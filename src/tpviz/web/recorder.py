"""Record a scene's choreography as keyframe tracks, without rendering.

RecorderScene overrides play()/wait(): animations are compiled with manim's
own `compile_animations`, begun, and SAMPLED at five wall-time fractions
(rate functions and lag ratios apply inside `Animation.interpolate`, so the
samples capture easing, `path_arc` flights, and Indicate's there-and-back as
piecewise-linear values). Scene-membership diffs give spawn/despawn times —
ReplacementTransform et al. become crossfades automatically. The existing
scenes remain the single source of truth for what happens on screen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from manim import Mobject, Scene

from tpviz.web import serializers

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
N = len(ALPHAS)


@dataclass
class ObjRecord:
    oid: int
    spec: dict  # class + draw params from serializers.serialize()
    t0: float
    t1: float | None = None
    tracks: dict[str, list] = field(default_factory=dict)  # prop -> flat segment list


@dataclass
class StepSeg:
    step: object
    t0: float
    t1: float


@dataclass
class Recording:
    objects: list[ObjRecord]
    steps: list[StepSeg]
    sections: list[tuple[str, float, float]]
    duration: float


class RecorderMixin:
    """Mix in FRONT of a concrete scene class. Never call render()."""

    def rec_init(self):
        self.clock = 0.0
        self._units: dict[int, ObjRecord] = {}  # id(mobject) -> record
        self._live: dict[int, Mobject] = {}
        # pin every recorded mobject: CPython reuses id() of collected objects,
        # and a fresh mobject at a dead one's address would silently never
        # register (this exact bug ate the equation strip)
        self._pins: list[Mobject] = []
        self._steps: list[StepSeg] = []
        self._sections: list[tuple[str, float, float]] = []
        self._cur_section = "setup"
        self._sec_t0 = 0.0

    # ------------------------------------------------------------ scene API
    def next_section(self, name="", *a, **k):
        self._sections.append((self._cur_section, self._sec_t0, self.clock))
        self._cur_section, self._sec_t0 = name or "section", self.clock

    def wait(self, duration=1.0, **k):
        self._sync_membership()
        self.clock += duration

    def play_step(self, step):
        t0 = self.clock
        super().play_step(step)
        self._steps.append(StepSeg(step, t0, self.clock))

    def mark_step(self, step):  # PP scenes call this around their custom loops
        self._pending_mark = (step, self.clock)

    def flush_mark(self):
        if getattr(self, "_pending_mark", None) is not None:
            step, t0 = self._pending_mark
            self._steps.append(StepSeg(step, t0, self.clock))
            self._pending_mark = None

    def play(self, *args, **kwargs):
        self._sync_membership()
        anims = self.compile_animations(*args, **kwargs)
        run_time = kwargs.get("run_time") or max(
            (getattr(a, "run_time", 1.0) for a in anims), default=1.0
        )
        self.add_mobjects_from_animations(anims)  # manim's own pre-play bookkeeping
        for a in anims:
            if "run_time" in kwargs:
                a.run_time = kwargs["run_time"]
            a._setup_scene(self)
            a.begin()
        self._sync_membership()  # introducers added their mobjects at clock t0

        samples: list[dict[int, dict]] = []
        for alpha in ALPHAS:
            for a in anims:
                a_alpha = min(1.0, alpha * run_time / max(a.run_time, 1e-9))
                a.interpolate(a_alpha)
            samples.append(self._snapshot())
        for a in anims:
            a.finish()
            a.clean_up_from_scene(self)

        t0, t1 = self.clock, self.clock + run_time
        self._emit(samples, t0, t1)
        self.clock = t1
        self._sync_membership()  # removals from clean_up (FadeOut, RT sources)

    # ------------------------------------------------------------ recording
    def _leaf_units(self) -> dict[int, Mobject]:
        out: dict[int, Mobject] = {}
        for top in self.mobjects:
            for m in serializers.iter_units(top):
                out[id(m)] = m
        return out

    def _sync_membership(self):
        live = self._leaf_units()
        for oid, m in live.items():
            if oid not in self._units:
                spec = serializers.serialize(m)
                if spec is None:
                    continue
                self._units[oid] = ObjRecord(oid=oid, spec=spec, t0=self.clock)
                self._live[oid] = m
                self._pins.append(m)
        for oid in list(self._live):
            if oid not in live:
                rec = self._units[oid]
                if rec.t1 is None:
                    rec.t1 = self.clock
                del self._live[oid]

    def _snapshot(self) -> dict[int, dict]:
        return {oid: serializers.state(m) for oid, m in self._live.items()}

    def _emit(self, samples: list[dict[int, dict]], t0: float, t1: float):
        for oid in samples[-1].keys() | samples[0].keys():
            rec = self._units.get(oid)
            if rec is None:
                continue
            per_prop: dict[str, list] = {}
            for prop in ("x", "y", "w", "o"):
                vals = []
                for snap in samples:
                    st = snap.get(oid)
                    if st is None:
                        vals = None
                        break
                    vals.append(st[prop])
                if vals is None:
                    continue
                if max(vals) - min(vals) > (0.004 if prop != "o" else 0.01):
                    per_prop[prop] = vals
            for prop, vals in per_prop.items():
                rec.tracks.setdefault(prop, []).append((t0, t1, vals))

    # ------------------------------------------------------------- results
    def recording(self) -> Recording:
        self._sync_membership()
        self._sections.append((self._cur_section, self._sec_t0, self.clock))
        for rec in self._units.values():
            if rec.t1 is None:
                rec.t1 = self.clock
        return Recording(
            objects=list(self._units.values()),
            steps=self._steps,
            sections=self._sections,
            duration=self.clock,
        )


def record(scene_cls: type[Scene]) -> tuple[Recording, object]:
    """Run scene_cls.construct() under the recorder; returns (recording, scene)."""
    rec_cls = type(f"Rec{scene_cls.__name__}", (RecorderMixin, scene_cls), {})
    scene = rec_cls()
    scene.rec_init()
    scene.construct()
    return scene.recording(), scene
