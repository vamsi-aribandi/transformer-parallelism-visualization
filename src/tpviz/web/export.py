"""Export CLI: record scenes -> quantized JSON -> self-contained web page.

    uv run python -m tpviz.web.export --only tp_fwd,tp_train
    uv run python -m tpviz.web.export --json-only

Writes web/dist/data.json always; inlines data + JS + CSS into
web/dist/index.html when the template exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tpviz import configs
from tpviz.core.backward import train_steps
from tpviz.core.model import forward_steps
from tpviz.web import serializers
from tpviz.web.recorder import record
from tpviz.web.schema import DocumentBuilder, validate

ROOT = Path(__file__).resolve().parents[3]
WEB = ROOT / "web"
DIST = WEB / "dist"

STRATEGY_LABEL = {
    "dp": "Data Parallelism", "fsdp": "FSDP (ZeRO-3)", "tp": "Tensor Parallelism",
    "cp": "Context Parallelism", "pp": "Pipeline Parallelism", "ep": "Expert Parallelism",
    "5d": "5D Parallelism",
}


def scene_registry() -> dict[str, tuple[type, object, bool]]:
    from tpviz.scenes.cp import CPScene
    from tpviz.scenes.cp_train import CPTrainScene
    from tpviz.scenes.dp import DPScene
    from tpviz.scenes.dp_train import DPTrainScene
    from tpviz.scenes.ep import EPScene
    from tpviz.scenes.ep_train import EPTrainScene
    from tpviz.scenes.fsdp import FSDPScene
    from tpviz.scenes.fsdp_train import FSDPTrainScene
    from tpviz.scenes.pp import PPScene
    from tpviz.scenes.pp_train import PPTrainScene
    from tpviz.scenes.tp import TPScene
    from tpviz.scenes.tp_train import TPTrainScene

    reg = {}
    for short, fwd_cls, train_cls in (
        ("dp", DPScene, DPTrainScene),
        ("fsdp", FSDPScene, FSDPTrainScene),
        ("tp", TPScene, TPTrainScene),
        ("cp", CPScene, CPTrainScene),
        ("pp", PPScene, PPTrainScene),
        ("ep", EPScene, EPTrainScene),
    ):
        cfg = getattr(configs, short.upper())
        reg[f"{short}_fwd"] = (fwd_cls, cfg, False)
        reg[f"{short}_train"] = (train_cls, cfg, True)
    from tpviz.scenes.five_d import FiveDScene

    reg["5d_train"] = (FiveDScene, configs.FIVE_D, True)
    return reg


def build_document(only: list[str] | None = None) -> dict:
    reg = scene_registry()
    names = only or list(reg)
    builder = DocumentBuilder()
    meta = {"strategies": {}, "modes": {}}
    for name in names:
        scene_cls, cfg, is_train = reg[name]
        serializers.UNSERIALIZED.clear()
        rec, _scene = record(scene_cls)
        if serializers.UNSERIALIZED:
            raise RuntimeError(f"{name}: unserialized mobjects: {set(serializers.UNSERIALIZED)}")
        semantic = train_steps(cfg) if is_train else forward_steps(cfg)
        builder.add_timeline(name, rec, semantic)
        short = name.rsplit("_", 1)[0]
        from tpviz.web.texthtml import tex_to_html

        meta["strategies"][short] = {
            "label": STRATEGY_LABEL[short],
            "name": cfg.name,
            "tagline": cfg.tagline,
            "meshh": tex_to_html(cfg.mesh.tex()),
            # mesh axes with the parallelism each implements and its size —
            # the figure's attribution chips and the wire-level inset use them
            "axes": {
                ax: {"role": cfg.axis_roles.get(ax, ""), "n": n}
                for ax, n in cfg.mesh.axes.items()
            },
        }
        meta["modes"][name] = "train" if is_train else "fwd"
    return builder.build(meta)


def merge_documents(base: dict, new: dict) -> dict:
    """Replace `new`'s timelines inside `base` (a previous full export),
    re-keying the shared glyph atlas (ids are positional) and tooltip indices."""
    out = json.loads(json.dumps(base))
    by_d = {d: gid for gid, d in out["glyphs"].items()}
    gmap = {}
    for gid, d in new["glyphs"].items():
        if d not in by_d:
            by_d[d] = f"G{len(out['glyphs'])}"
            out["glyphs"][by_d[d]] = d
        gmap[gid] = by_d[d]
    for key, eq in new["eq"].items():
        if key not in out["eq"]:
            out["eq"][key] = {**eq, "u": [[gmap[g], x, y] for g, x, y in eq["u"]]}
    tip_index = {json.dumps(t, sort_keys=True): i for i, t in enumerate(out["tooltips"])}
    tmap = {}
    for i, t in enumerate(new["tooltips"]):
        k = json.dumps(t, sort_keys=True)
        if k not in tip_index:
            tip_index[k] = len(out["tooltips"])
            out["tooltips"].append(t)
        tmap[i] = tip_index[k]
    for name, tl in new["timelines"].items():
        tl = json.loads(json.dumps(tl))
        for o in tl["objects"]:
            if "tip" in o:
                o["tip"] = tmap[o["tip"]]
        out["timelines"][name] = tl
    for k in ("strategies", "modes"):
        out["meta"].setdefault(k, {}).update(new["meta"].get(k, {}))
    out["dims"] = new["dims"]
    return out


def write_embed_data(doc: dict) -> None:
    """dist/data.js: the document as a window global, for embedding the figures
    in a host site (script src) instead of the standalone page's inline tag."""
    payload = json.dumps(doc, separators=(",", ":")).replace("</", "<\\/")
    (DIST / "data.js").write_text(f"window.TPVIZ_DATA={payload};")


def build_page(doc: dict) -> str | None:
    template = WEB / "article.html"
    if not template.exists():
        return None
    html = template.read_text()
    css = (WEB / "style.css").read_text()
    js = (WEB / "tpviz.js").read_text()
    payload = json.dumps(doc, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("<!--DATA-->", f'<script type="application/json" id="tpviz-data">{payload}</script>')
    html = html.replace("<!--CSS-->", f"<style>{css}</style>")
    html = html.replace("<!--JS-->", f"<script>{js}</script>")
    return html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated timeline names, e.g. tp_fwd,tp_train "
                                   "(merged into the existing dist/data.json)")
    ap.add_argument("--json-only", action="store_true")
    ap.add_argument("--page-only", action="store_true",
                    help="skip recording: rebuild index.html from the existing dist/data.json")
    args = ap.parse_args()
    only = args.only.split(",") if args.only else None

    if args.page_only:
        doc = json.loads((DIST / "data.json").read_text())
        page = build_page(doc)
        (DIST / "index.html").write_text(page)
        write_embed_data(doc)
        print(f"index.html: {(DIST / 'index.html').stat().st_size / 1e6:.2f} MB (from existing data.json)")
        return

    doc = build_document(only)
    if only and (DIST / "data.json").exists():
        # keep the other timelines from the previous full export
        doc = merge_documents(json.loads((DIST / "data.json").read_text()), doc)
    errs = validate(doc)
    if errs:
        raise SystemExit("validation failed:\n" + "\n".join(errs[:20]))

    DIST.mkdir(parents=True, exist_ok=True)
    data_path = DIST / "data.json"
    data_path.write_text(json.dumps(doc, separators=(",", ":")))
    write_embed_data(doc)
    size = data_path.stat().st_size
    print(f"data.json: {size / 1e6:.2f} MB, timelines: {list(doc['timelines'])}")
    assert size < 8_000_000, "document over budget"

    if not args.json_only:
        page = build_page(doc)
        if page is not None:
            (DIST / "index.html").write_text(page)
            print(f"index.html: {(DIST / 'index.html').stat().st_size / 1e6:.2f} MB")
        else:
            print("template.html missing — wrote data.json only")


if __name__ == "__main__":
    main()
