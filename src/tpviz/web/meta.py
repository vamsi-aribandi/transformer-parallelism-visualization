"""Program-panel entries and tensor tooltips (concrete dims, memory)."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from tpviz.core.steps import CollectiveStep, MatMulStep, P2PSendStep, SaveActivationStep
from tpviz.scenes.base import default_caption
from tpviz.web.texthtml import tex_to_html


@dataclass(frozen=True)
class NominalDims:
    B: int = 8
    T: int = 128
    D: int = 1024
    F: int = 4096
    H: int = 1024
    E: int = 4
    S: int = 256  # B*T/E, balanced top-1 routing
    bytes_per: int = 2  # bf16

    def size(self, dim: str) -> int:
        return getattr(self, dim, 1)


NOMINAL = NominalDims()


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def tensor_tooltip(tensor: dict, atlas) -> dict:
    """Build tooltip payload from a serialized tensor spec dict."""
    from tpviz.core.mesh import Mesh

    dims = tensor["dims"]
    shard = tensor["shard"]
    mesh = Mesh(tensor["mesh"])
    device = tensor["device"]
    # E follows the expert axis of the mesh (4 experts in the EP single, 2 in
    # the 5D combo); S = B*T/E is the balanced routed-token count
    sizes = {d: NOMINAL.size(d) for d in dims}
    if "E" in sizes:
        sizes["E"] = mesh.axes.get("Z", NOMINAL.E)
        sizes["S"] = NOMINAL.B * NOMINAL.T // sizes["E"]
    glob = [sizes[d] for d in dims]
    local = [sizes[d] // (mesh.size(shard[d]) if d in shard else 1) for d in dims]
    mem_g = prod(glob) * NOMINAL.bytes_per
    mem_l = prod(local) * NOMINAL.bytes_per
    sharding_axes = [a for axes in shard.values() for a in axes]
    others = [a for a in mesh.axes if a != "stage" and a not in sharding_axes]
    if shard:
        parts = []
        for d, ax in shard.items():
            n = mesh.size(ax)
            size = sizes[d] // n
            lo = mesh.coord(ax, device) * size
            parts.append(f"{d} sharded {n} ways over {ax}; this device holds {d}∈[{lo}, {lo + size})")
        shard_desc = "; ".join(parts)
        if others and len(mesh.axes) > 1:
            shard_desc += f"; replicated over {', '.join(others)}"
    else:
        n = prod(mesh.axes.values()) if mesh.axes else 1
        shard_desc = f"replicated on all {n} devices"
    return {
        "kind": tensor["kind"],
        "global": glob,
        "local": local,
        "dims": dims,
        "memGlobal": _fmt_bytes(mem_g),
        "memLocal": _fmt_bytes(mem_l),
        "shardDesc": shard_desc,
        "partial": tensor["partial"] or None,
    }


def step_entry(seg, atlas) -> dict:
    """One program-panel entry from a recorded StepSeg."""
    step = seg.step
    kind = type(step).__name__.removesuffix("Step")
    entry = {
        "t0": seg.t0,
        "t1": seg.t1,
        "kind": kind,
        "layer": step.layer,
        "phase": step.phase,
        "bwd": step.backward,
        "cap": step.caption or default_caption(step),
    }
    if step.microbatch is not None:
        entry["mb"] = step.microbatch
    if kind == "StageCompute":
        # PP scenes mark one step per tick and caption it with every stage's
        # work — that caption IS the program line
        entry["eqh"] = step.caption or (
            f"stage {step.stage} · layer {step.layer} · "
            f"{'backward' if step.backward else 'forward'} mb{step.microbatch}"
        )
        entry["cap"] = None
    elif hasattr(step, "tex"):
        entry["eqh"] = tex_to_html(step.tex())
    elif isinstance(step, SaveActivationStep):
        entry["eqh"] = tex_to_html(step.t.tex())
        entry["save"] = True
    else:
        entry["eqh"] = entry["kind"]
    if step.note_tex:
        entry["noteh"] = tex_to_html(step.note_tex)
    # attribution: which mesh axis (= which parallelism) this collective belongs to
    if isinstance(step, P2PSendStep):
        entry["axis"] = "stage"
    elif isinstance(step, CollectiveStep):
        entry["axis"] = step.axis
    entry["comm"] = isinstance(step, (CollectiveStep, P2PSendStep))
    entry["mm"] = isinstance(step, MatMulStep)
    return entry
