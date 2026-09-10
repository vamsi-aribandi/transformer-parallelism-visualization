"""The six single-strategy configurations.

Axis-name conventions (kept distinct so future multi-axis combos compose):
  X = data / FSDP / context axis, Y = tensor axis, Z = expert axis,
  'stage' = pipeline stages (not a sharding subscript — stages own whole layers).
"""

from __future__ import annotations

from tpviz.core.mesh import Mesh, MoEConfig, PipelineConfig, StrategyConfig

DP = StrategyConfig(
    name="Data Parallelism",
    short="dp",
    tagline="Shard the batch. Replicate the weights.",
    mesh=Mesh({"X": 4}),
    act_sharding={"B": "X"},
    wt_sharding={},
)

FSDP = StrategyConfig(
    name="Fully-Sharded Data Parallelism (ZeRO-3)",
    short="fsdp",
    tagline="Shard the weights too. Gather them just-in-time.",
    mesh=Mesh({"X": 4}),
    act_sharding={"B": "X"},
    wt_sharding={
        "W_qkv": {"D": "X"},
        "W_o": {"D": "X"},
        "W_in": {"D": "X"},
        "W_out": {"D": "X"},
    },
)

TP = StrategyConfig(
    name="Tensor Parallelism (Megatron)",
    short="tp",
    tagline="Shard the features and heads. AllGather in, ReduceScatter out.",
    mesh=Mesh({"Y": 4}),
    act_sharding={"D": "Y"},
    wt_sharding={
        "W_qkv": {"H": "Y"},
        "W_o": {"H": "Y"},
        "W_in": {"F": "Y"},
        "W_out": {"F": "Y"},
    },
)

CP = StrategyConfig(
    name="Context Parallelism",
    short="cp",
    tagline="Shard the sequence. Only attention needs to talk.",
    mesh=Mesh({"X": 4}),
    act_sharding={"T": "X"},
    wt_sharding={},
    kv_context_axis="X",
)

PP = StrategyConfig(
    name="Pipeline Parallelism",
    short="pp",
    tagline="Shard the layers. Keep every stage busy with microbatches.",
    mesh=Mesh({"stage": 2}),
    pipeline=PipelineConfig(n_stages=2, n_microbatches=4),
)

EP = StrategyConfig(
    name="Expert Parallelism (MoE)",
    short="ep",
    tagline="Shard the experts. Route tokens with AllToAll.",
    mesh=Mesh({"Z": 4}),
    act_sharding={"B": "Z"},
    wt_sharding={},
    moe=MoEConfig(n_experts=4, axis="Z"),
)

ALL = {c.short: c for c in (DP, FSDP, TP, CP, PP, EP)}
