"""Device mesh and per-strategy configuration.

Sharding subscripts are strings of single-letter mesh axes: "X" (one axis) or
"XZ" (a dim split over two axes at once, e.g. the batch under FSDP x EP).
`AXIS_ORDER` fixes the letter order inside such compound subscripts so that
`B_{XZ}` and `{U_{XZC}}` always read the same way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from tpviz import notation

AXIS_ORDER = "XYZC"  # canonical order of letters inside a compound subscript


def canon_axes(axes: str) -> str:
    """Deduplicate + order the letters of a compound sharding subscript."""
    return "".join(sorted(set(axes), key=lambda a: (AXIS_ORDER.find(a) % 99, a)))


@dataclass(frozen=True)
class Mesh:
    axes: dict[str, int]  # e.g. {"X": 4}; combos: {"X": 2, "Y": 2, "C": 2, "Z": 2, "stage": 2}

    @property
    def n_devices(self) -> int:
        return math.prod(self.axes.values())

    def size(self, axes: str) -> int:
        """Device count along `axes` — one axis name, or a compound of letters."""
        if axes in self.axes:
            return self.axes[axes]
        return math.prod(self.axes[a] for a in axes)

    def coords(self, device: int) -> dict[str, int]:
        """Flat device index -> per-axis coordinates (row-major over `axes`)."""
        out: dict[str, int] = {}
        rem = device
        for name, size in reversed(list(self.axes.items())):
            out[name] = rem % size
            rem //= size
        return out

    def coord(self, axes: str, device: int) -> int:
        """Coordinate of `device` along `axes`: a single axis, or the row-major
        combined coordinate over the letters of a compound subscript."""
        c = self.coords(device)
        if axes in self.axes:
            return c[axes]
        idx = 0
        for a in axes:
            idx = idx * self.axes[a] + c[a]
        return idx

    def index(self, **coords: int) -> int:
        """Per-axis coordinates -> flat device index."""
        idx = 0
        for name, size in self.axes.items():
            idx = idx * size + coords.get(name, 0)
        return idx

    def tex(self) -> str:
        return notation.mesh_tex(self.axes)


@dataclass(frozen=True)
class PipelineConfig:
    n_stages: int
    n_microbatches: int
    # composed=True: stage s runs layer s+1's ops INLINE (with all of the other
    # axes' collectives) and a P2P send joins the layers — used by the
    # multi-axis combos. False: the GPipe microbatch schedule of the PP single.
    composed: bool = False


@dataclass(frozen=True)
class MoEConfig:
    n_experts: int
    axis: str  # mesh axis the expert dim E is sharded over


DimSharding = dict[str, str]  # logical dim -> mesh axis letters; absent = replicated


@dataclass(frozen=True)
class StrategyConfig:
    name: str  # "Data Parallelism (DP)"
    short: str  # "dp" — filenames / scene ids
    tagline: str  # one-liner for the title card
    mesh: Mesh
    act_sharding: DimSharding = field(default_factory=dict)  # sharding of In[B, T, D]
    wt_sharding: dict[str, DimSharding] = field(default_factory=dict)  # per weight name
    kv_context_axis: str | None = None  # CP: axis sharding T -> AllGather K, V in attention
    pipeline: PipelineConfig | None = None
    moe: MoEConfig | None = None
    prefer_reduce_scatter: bool = True  # TP: ReduceScatter (vs AllReduce) on partial outputs
    n_layers: int = 2
    # which parallelism each mesh axis implements — the attribution shown on
    # every collective ("AllGather over X · FSDP")
    axis_roles: dict[str, str] = field(default_factory=dict)
    # ZeRO-1: weights stay replicated over this data axis but gradients
    # ReduceScatter onto it, each device updates its shard, and the updated
    # weights AllGather afterwards (an AllReduce split around the optimizer)
    zero1_axis: str | None = None
