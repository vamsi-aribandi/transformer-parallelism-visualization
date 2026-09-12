"""Assemble recordings into the DISCRETE step-state document the article
figures consume.

Model: navigation is by STEP, not by time. For each step we store
  - `d`:    exact sparse diff of object states at the step's end boundary
            (teleport target — scrubbing snaps here, no interpolation),
  - `in`/`gone`: objects that spawn/despawn by this boundary,
  - `beats`: the step's constituent plays as short tweens (used only in
            play mode; the player snaps to `d` afterwards so drift is
            impossible).
Everything before the first step (title card, initial placement) is folded
into the figure's base state; everything after the last step (the video's
summary card) is dropped — the article's prose carries that.

Quantization: coordinates in integer centi-scene-units (y-up, client flips),
width as ratio-to-spawn x100, opacity x100, beat durations in ms.
"""

from __future__ import annotations

import json

from tpviz import style
from tpviz.core.backward import count_matmuls
from tpviz.core.model import count_collectives
from tpviz.web.meta import step_entry, tensor_tooltip, NOMINAL
from tpviz.web.recorder import Recording
from tpviz.web.texatlas import TexAtlas

CS = 100
STRIP_Z = style.Z_STRIP  # recorded equation-strip objects are replaced by HTML

# baked hexes -> theme tokens the client resolves to CSS variables
HEX2TOK: dict[str, str] = {}
for _hex, _tok in [
    (style.TEXT_COLOR, "text"), (style.MUTED_TEXT, "muted"), (style.ACCENT, "accent"),
    (style.COMM_COLOR, "comm"), (style.GOOD_COLOR, "good"),
    (style.ACT_FILL, "act"), (style.ACT_STROKE, "actS"),
    (style.WEIGHT_FILL, "wt"), (style.WEIGHT_STROKE, "wtS"),
    (style.KV_FILL, "kv"), (style.KV_STROKE, "kvS"),
    (style.GRAD_FILL, "grad"), (style.GRAD_STROKE, "gradS"),
    (style.DEVICE_BOX_FILL, "boxFill"), (style.DEVICE_BOX_STROKE, "boxStroke"),
    ("#151b24", "surf"), ("#1a2029", "surf2"), ("#243041", "surf2"),
    ("#e5e7eb", "text"),
]:
    HEX2TOK[_hex.upper()] = _tok
for _i, _hex in enumerate(style.DEVICE_HUES[:4]):
    HEX2TOK[_hex.upper()] = f"dev{_i}"
for _i, _hex in enumerate(style.EXPERT_HUES):
    HEX2TOK[_hex.upper()] = f"exp{_i}"


def _tok(color):
    if isinstance(color, str):
        return HEX2TOK.get(color.upper(), color)
    return color


def _q(v: float) -> int:
    return round(v * CS)


def _arc_samples(o, a: float, b: float) -> list | None:
    """Interior (x, y) samples for a play interval whose path visibly bends
    (collective crossflies with path_arc) — restores the arc in play mode."""
    xs = ys = None
    for seg in o.tracks.get("x", []):
        if abs(seg[0] - a) < 1e-6 and abs(seg[1] - b) < 1e-6:
            xs = seg[2]
    for seg in o.tracks.get("y", []):
        if abs(seg[0] - a) < 1e-6 and abs(seg[1] - b) < 1e-6:
            ys = seg[2]
    if not xs and not ys:
        return None
    n = max(len(xs) if xs else 0, len(ys) if ys else 0)
    if n < 3:
        return None
    xs = xs or [o.spec["x0"]] * n
    ys = ys or [o.spec["y0"]] * n
    if len(xs) != n or len(ys) != n:
        return None
    dev = 0.0
    for k in range(1, n - 1):
        u = k / (n - 1)
        dev = max(
            dev,
            abs(xs[k] - (xs[0] + (xs[-1] - xs[0]) * u)),
            abs(ys[k] - (ys[0] + (ys[-1] - ys[0]) * u)),
        )
    if dev < 0.06:
        return None
    out = []
    for k in range(1, n - 1):
        out += [_q(xs[k]), _q(ys[k])]
    return out


def _sample(segs: list, t: float, fallback: float) -> float:
    """Value of a track at time t (same math as the old player, python-side)."""
    best = None
    for seg in segs:
        if seg[0] <= t:
            best = seg
        else:
            break
    if best is None:
        return fallback
    t0, t1, vals = best
    if t >= t1:
        return vals[-1]
    u = (t - t0) / (t1 - t0) * (len(vals) - 1)
    i = min(int(u), len(vals) - 2)
    return vals[i] + (vals[i + 1] - vals[i]) * (u - i)


class DocumentBuilder:
    def __init__(self):
        self.atlas = TexAtlas()
        self.tooltips: list[dict] = []
        self._tip_index: dict[str, int] = {}
        self.timelines: dict[str, dict] = {}

    def _tip(self, tensor: dict) -> int:
        key = json.dumps(tensor, sort_keys=True)
        if key not in self._tip_index:
            self._tip_index[key] = len(self.tooltips)
            self.tooltips.append(tensor_tooltip(tensor, self.atlas))
        return self._tip_index[key]

    def add_timeline(self, name: str, rec: Recording, steps_semantic: list) -> None:
        if not rec.steps:
            raise ValueError(f"{name}: no recorded steps")
        # article time range: first content section start .. last step end
        t_base = min(
            (a for n, a, b in rec.sections if n not in ("setup", "title") and b > a),
            default=rec.steps[0].t0,
        )
        t_end = rec.steps[-1].t1

        # objects alive in the window, minus the recorded equation strip
        keep = [
            o for o in rec.objects
            if o.t1 > t_base and o.t0 < t_end and o.spec.get("z", 0) < STRIP_Z
        ]
        index_of = {o.oid: i for i, o in enumerate(keep)}

        def obj_state(o, t):
            w0 = max(o.spec.get("w0", 1e-4), 1e-4)
            return (
                _q(_sample(o.tracks.get("x", []), t, o.spec["x0"])),
                _q(_sample(o.tracks.get("y", []), t, o.spec["y0"])),
                round(_sample(o.tracks.get("w", []), t, w0) / w0 * 100),
                round(_sample(o.tracks.get("o", []), t, o.spec["o0"]) * 100),
            )

        objects = []
        for o in keep:
            spec = dict(o.spec)
            entry = {
                "c": spec.pop("c"),
                "w": _q(spec.pop("w0")),
                "h": _q(spec.pop("h0")),
                "z": spec.pop("z", 0),
            }
            for k in ("x0", "y0", "o0"):
                spec.pop(k, None)
            if "tensor" in spec:
                tensor = spec.pop("tensor")
                entry["tip"] = self._tip(tensor)
                entry["tex"] = self.atlas.add(spec.pop("tex"))
                entry["tensor"] = tensor
            if "tex" in spec:
                entry["eq"] = self.atlas.add(spec.pop("tex"))
            for k in ("color", "fill", "stroke"):
                if k in spec:
                    spec[k] = _tok(spec[k])
            entry.update(spec)
            objects.append(entry)
        save_objs = sorted(i for i, o in enumerate(objects) if o.pop("sv", None))

        alive = {i: False for i in range(len(keep))}
        state = {}

        def boundary(t):
            """(diff-vs-state, spawned, despawned) at time t; updates state."""
            diff, born, gone = [], [], []
            for i, o in enumerate(keep):
                now_alive = o.t0 <= t and (o.t1 > t or o.t1 >= t_end)
                if now_alive:
                    st = obj_state(o, t)
                    if not alive[i]:
                        born.append(i)
                        diff.append([i, *st])
                    elif state.get(i) != st:
                        diff.append([i, *st])
                    state[i] = st
                elif alive[i]:
                    gone.append(i)
                alive[i] = now_alive
            return diff, born, gone

        base_diff, base_born, _ = boundary(t_base)
        base = base_diff  # every entry at t_base is a full state row

        # beats: play intervals from track segments, grouped per step window
        play_ivals = sorted({
            (seg[0], seg[1])
            for o in keep for segs in o.tracks.values() for seg in segs
            if seg[1] > t_base and seg[0] < t_end
        })

        entries = []
        prev_end = t_base
        coll = mm = 0
        for seg_step in rec.steps:
            e = step_entry(seg_step, self.atlas)
            window = (prev_end, seg_step.t1)
            beats = []
            snapshot = dict(state)
            live = dict(alive)
            for (a, b) in play_ivals:
                if a < window[0] or b > window[1] + 1e-6:
                    continue
                ch, born, gone, sp = [], [], [], []
                for i, o in enumerate(keep):
                    now = o.t0 <= b and (o.t1 > b or o.t1 >= t_end)
                    if now:
                        st = obj_state(o, b)
                        if not live[i]:
                            born.append(i)
                            sp.append([i, *obj_state(o, a)])
                        if snapshot.get(i) != st or not live[i]:
                            row = [i, *st]
                            arc = _arc_samples(o, a, b)
                            if arc:
                                row.append(arc)
                            ch.append(row)
                            snapshot[i] = st
                    elif live[i]:
                        gone.append(i)
                    live[i] = now
                if ch or born or gone:
                    beats.append({
                        "d": max(150, min(900, round((b - a) * 10))),  # cs -> ms
                        "ch": ch, "in": born, "out": gone, "sp": sp,
                    })
            diff, born, gone = boundary(seg_step.t1)
            coll += e.pop("comm")
            mm += e.pop("mm")
            e.pop("t0", None)
            e.pop("t1", None)
            e.update({"d": diff, "in": born, "gone": gone, "beats": beats,
                      "coll": coll, "mm": mm})
            entries.append(e)
            prev_end = seg_step.t1

        fwd_tex = {}
        for i, seg_step in enumerate(rec.steps):
            st = seg_step.step
            if not st.backward and hasattr(st, "tex"):
                fwd_tex.setdefault(st.tex(), i)
        for i, seg_step in enumerate(rec.steps):
            st = seg_step.step
            if st.backward and st.note_tex:
                for tex, k in fwd_tex.items():
                    if st.note_tex.endswith(tex):
                        entries[i]["fwdStep"] = k
                        break

        mm_counts = count_matmuls(steps_semantic)
        self.timelines[name] = {
            "objects": objects,
            "base": base,
            "baseIn": base_born,
            "steps": entries,
            "saveObjs": save_objs,
            "summary": {"coll": count_collectives(steps_semantic), "mm": mm_counts},
        }

    def build(self, meta: dict | None = None) -> dict:
        return {
            "v": 2,
            "frame": [1422, 800],
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
        seen = set(i for i, *_ in tl["base"])
        for si, st in enumerate(tl["steps"]):
            for row in st["d"]:
                if not (0 <= row[0] < n):
                    errs.append(f"{name}: step {si} diff references object {row[0]}")
            if "eqh" not in st:
                errs.append(f"{name}: step {si} missing eqh")
            seen.update(st["in"])
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
