"""Regenerate program captions, notes and tooltip text in dist/data.json from
the current code, without re-recording. Each recorded step entry is matched
(in order, by its equation) to the semantic step list; its caption and note
are rebuilt from that step. Tooltips are rebuilt from their tensor specs.

    uv run python scripts/rewording.py
"""

import json
from pathlib import Path

from tpviz.core.backward import train_steps
from tpviz.core.steps import SaveActivationStep
from tpviz.scenes.base import default_caption
from tpviz.web.export import scene_registry
from tpviz.web.meta import tensor_tooltip
from tpviz.web.texthtml import tex_to_html

DATA = Path(__file__).resolve().parents[1] / "web" / "dist" / "data.json"


def main():
    doc = json.loads(DATA.read_text())
    reg = scene_registry()
    grid = {"5d_train", "dense4d_train", "moe4d_train"}
    changed = 0
    for name, tl in doc["timelines"].items():
        _cls, cfg, _ = reg[name]
        sem = train_steps(cfg)
        if name in grid:
            from tpviz.scenes.five_d import attribution_caption
            sem = [s for s in sem if not isinstance(s, SaveActivationStep)]
            cap_of = lambda s: attribution_caption(cfg, s)  # noqa: E731
        else:
            cap_of = lambda s: s.caption or default_caption(s)  # noqa: E731
        k = 0
        for e in tl["steps"]:
            if e["kind"] == "StageCompute":
                continue  # PP tick lines are captioned by the scene itself
            for j in range(k, len(sem)):
                st = sem[j]
                eqh = tex_to_html(st.t.tex()) if isinstance(st, SaveActivationStep) else (
                    tex_to_html(st.tex()) if hasattr(st, "tex") else None)
                if eqh == e["eqh"] and st.backward == e["bwd"]:
                    new_cap = cap_of(st)
                    new_note = tex_to_html(st.note_tex) if st.note_tex else None
                    if e.get("cap") != new_cap or e.get("noteh") != new_note:
                        changed += 1
                    e["cap"] = new_cap
                    if new_note:
                        e["noteh"] = new_note
                    else:
                        e.pop("noteh", None)
                    k = j + 1
                    break
    # tooltips: need their source tensor spec, which objects carry
    specs = {}
    for tl in doc["timelines"].values():
        for o in tl["objects"]:
            if "tip" in o and "tensor" in o:
                specs.setdefault(o["tip"], o["tensor"])
    for i, spec in specs.items():
        doc["tooltips"][i] = tensor_tooltip(spec, None)
    DATA.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"{changed} step entries reworded, {len(specs)} tooltips rebuilt")


if __name__ == "__main__":
    main()
