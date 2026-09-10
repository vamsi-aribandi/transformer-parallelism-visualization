"""Device mesh and per-strategy configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from tpviz import notation


@dataclass(frozen=True)
class Mesh:
    axes: dict[str, int]  # e.g. {"X": 4}; later combos: {"X": 2, "Y": 2, ...}

    @property
    def n_devices(self) -> int:
        return math.prod(self.axes.values())

    def size(self, axis: str) -> int:
        return self.axes[axis]

    def tex(self) -> str:
        return notation.mesh_tex(self.axes)


@dataclass(frozen=True)
class PipelineConfig:
    n_stages: int
    n_microbatches: int


@dataclass(frozen=True)
class MoEConfig:
    n_experts: int
    axis: str  # mesh axis the expert dim E is sharded over


DimSharding = dict[str, str]  # logical dim -> mesh axis; absent = replicated


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
