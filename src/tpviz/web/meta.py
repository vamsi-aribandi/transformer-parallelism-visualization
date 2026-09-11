"""Program-panel entries and tensor tooltips (concrete dims, memory)."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from tpviz.core.steps import CollectiveStep, MatMulStep, P2PSendStep, SaveActivationStep
from tpviz.scenes.base import default_caption


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
    dims = tensor["dims"]
    shard = tensor["shard"]
    mesh = tensor["mesh"]
    device = tensor["device"]
    glob = [NOMINAL.size(d) for d in dims]
    local = [
        NOMINAL.size(d) // (mesh.get(shard[d], 1) if d in shard else 1) for d in dims
    ]
    mem_g = prod(glob) * NOMINAL.bytes_per
    mem_l = prod(local) * NOMINAL.bytes_per
    if shard:
        parts = []
        for d, ax in shard.items():
            n = mesh.get(ax, 1)
            size = NOMINAL.size(d) // n
            lo = (device % n) * size  # 1D meshes; fine for the singles
            parts.append(f"{d} sharded {n}-way over {ax} — this device holds {d}∈[{lo}, {lo + size})")
        shard_desc = "; ".join(parts)
    else:
        n = prod(mesh.values()) if mesh else 1
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
    if hasattr(step, "tex"):
        entry["eq"] = atlas.add(step.tex())
    elif isinstance(step, SaveActivationStep):
        entry["eq"] = atlas.add(rf"\text{{save}}\ {step.t.tex()}")
    if step.note_tex:
        entry["note"] = atlas.add(step.note_tex)
    entry["comm"] = isinstance(step, (CollectiveStep, P2PSendStep))
    entry["mm"] = isinstance(step, MatMulStep)
    return entry
