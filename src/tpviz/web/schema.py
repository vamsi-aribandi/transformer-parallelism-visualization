"""Assemble recordings into the quantized JSON document the player consumes.

Quantization: times in integer centiseconds; coordinates in integer
centi-scene-units (manim frame 14.22 x 8 -> x in [-711, 711], y in [-400, 400],
y-up like manim; the client flips); widths as ratio-to-spawn x100; opacity x100.
"""

from __future__ import annotations

import json

from tpviz import style
from tpviz.core.backward import count_matmuls
from tpviz.core.model import count_collectives
from tpviz.web.meta import NOMINAL, step_entry, tensor_tooltip
from tpviz.web.recorder import Recording
from tpviz.web.texatlas import TexAtlas

CS = 100  # centi-units


def _q(v: float) -> int:
    return round(v * CS)


class DocumentBuilder:
    def __init__(self):
        self.atlas = TexAtlas()
        self.tooltips: list[dict] = []
        self._tip_index: dict[str, int] = {}
        self.timelines: dict[str, dict] = {}

    def _tip(self, tensor: dict) -> int:
        key = json.dumps(tensor, sort_keys=True)
        if key not in self._tip_index:
            tip = tensor_tooltip(tensor, self.atlas)
            self._tip_index[key] = len(self.tooltips)
            self.tooltips.append(tip)
        return self._tip_index[key]

    def add_timeline(self, name: str, rec: Recording, steps_semantic: list) -> None:
        objects, tracks = [], []
        index_of = {}
        for rec_obj in rec.objects:
            spec = dict(rec_obj.spec)
            o = {
                "c": spec.pop("c"),
                "t0": _q(rec_obj.t0),
                "t1": _q(rec_obj.t1),
                "x": _q(spec.pop("x0")),
                "y": _q(spec.pop("y0")),
                "w": _q(spec.pop("w0")),
                "h": _q(spec.pop("h0")),
                "o": round(spec.pop("o0") * 100),
                "z": spec.pop("z", 0),
            }
            if "tensor" in spec:
                tensor = spec.pop("tensor")
                o["tip"] = self._tip(tensor)
                o["tex"] = self.atlas.add(spec.pop("tex"))
                o["tensor"] = tensor
            if "tex" in spec:
                o["eq"] = self.atlas.add(spec.pop("tex"))
            o.update(spec)
            index_of[rec_obj.oid] = len(objects)
            objects.append(o)

        for rec_obj in rec.objects:
            idx = index_of[rec_obj.oid]
            w0 = max(rec_obj.spec.get("w0", 1e-4), 1e-4)
            for prop, segs in rec_obj.tracks.items():
                for t0, t1, vals in segs:
                    if prop == "w":
                        q = [round(v / w0 * 100) for v in vals]
                    elif prop == "o":
                        q = [round(v * 100) for v in vals]
                    else:
                        q = [_q(v) for v in vals]
                    tracks.append([idx, prop, _q(t0), _q(t1), q])
        tracks.sort(key=lambda r: (r[0], r[1], r[2]))

        entries = [step_entry(seg, self.atlas) for seg in rec.steps]
        counters, coll, mm = [], 0, 0
        for e in entries:
            e["t0"], e["t1"] = _q(e["t0"]), _q(e["t1"])
            coll += e.pop("comm")
            mm += e.pop("mm")
            counters.append([e["t1"], coll, mm])

        mm_counts = count_matmuls(steps_semantic)
        self.timelines[name] = {
            "dur": _q(rec.duration),
            "objects": objects,
            "tracks": tracks,
            "steps": entries,
            "sections": [[n, _q(a), _q(b)] for n, a, b in rec.sections if b > a],
            "counters": counters,
            "summary": {"coll": count_collectives(steps_semantic), "mm": mm_counts},
        }

    def build(self, meta: dict | None = None) -> dict:
        return {
            "v": 1,
            "frame": [1422, 800],
            "palette": {
                "bg": style.BACKGROUND,
                "text": style.TEXT_COLOR,
                "muted": style.MUTED_TEXT,
                "accent": style.ACCENT,
                "comm": style.COMM_COLOR,
                "good": style.GOOD_COLOR,
                "grad": style.GRAD_FILL,
                "gradStroke": style.GRAD_STROKE,
                "fill": style.TENSOR_FILL,
                "stroke": style.TENSOR_STROKE,
                "device": style.DEVICE_HUES,
                "expert": style.EXPERT_HUES,
                "mb": style.MICROBATCH_HUES,
            },
            "dims": {
                "B": NOMINAL.B, "T": NOMINAL.T, "D": NOMINAL.D, "F": NOMINAL.F,
                "H": NOMINAL.H, "E": NOMINAL.E, "S": NOMINAL.S, "dtype": "bf16",
            },
            "meta": meta or {},
            **self.atlas.emit(),
            "tooltips": self.tooltips,
            "timelines": self.timelines,
        }


def validate(doc: dict) -> list[str]:
    errs = []
    for name, tl in doc["timelines"].items():
        n = len(tl["objects"])
        for row in tl["tracks"]:
            if not (0 <= row[0] < n):
                errs.append(f"{name}: track references object {row[0]} of {n}")
            if row[2] >= row[3]:
                errs.append(f"{name}: empty track segment {row}")
        for st in tl["steps"]:
            if st["t0"] >= st["t1"]:
                errs.append(f"{name}: step {st['kind']} empty segment")
            for k in ("eq", "note"):
                if k in st and st[k] not in doc["eq"]:
                    errs.append(f"{name}: step eq ref {st[k]} missing")
        for o in tl["objects"]:
            for k in ("eq", "tex"):
                if k in o and o[k] not in doc["eq"]:
                    errs.append(f"{name}: object eq ref missing")
            if "tip" in o and not (0 <= o["tip"] < len(doc["tooltips"])):
                errs.append(f"{name}: bad tooltip ref")
    for eq in doc["eq"].values():
        for u in eq["u"]:
            if u[0] not in doc["glyphs"]:
                errs.append(f"eq glyph ref {u[0]} missing")
    return errs
