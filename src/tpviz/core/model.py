"""forward_steps(cfg): the 2-layer transformer forward pass as a step list.

Matmul collectives are DERIVED by the engine from the sharding specs (the
book's four cases). Only the pieces that are not matmul-shaped are emitted
explicitly: the attention K/V gather for context parallelism, MoE routing and
its AllToAlls, and the pipeline microbatch schedule.

Logical dims: activations [B, T, D]; heads fused as H = n_heads * d_head;
MLP hidden F; expert dim E; routed-token dim S.
"""

from __future__ import annotations

from dataclasses import replace

from tpviz.core.engine import plan_matmul
from tpviz.core.mesh import StrategyConfig
from tpviz.core.steps import (
    AllGatherStep,
    AllToAllStep,
    AnnotateStep,
    AttentionCoreStep,
    GeluStep,
    MatMulStep,
    P2PSendStep,
    RouteStep,
    SplitQKVStep,
    StageComputeStep,
    Step,
)
from tpviz.core.tensors import LTensor

WEIGHT_DIMS: dict[str, tuple[str, ...]] = {
    "W_qkv": ("D", "H"),
    "W_o": ("H", "D"),
    "W_in": ("D", "F"),
    "W_out": ("F", "D"),
}

MOE_WEIGHT_DIMS: dict[str, tuple[str, ...]] = {
    "W_in": ("E", "D", "F"),
    "W_out": ("E", "F", "D"),
}


def make_weight(cfg: StrategyConfig, name: str) -> LTensor:
    moe_mlp = cfg.moe is not None and name in ("W_in", "W_out")
    dims = MOE_WEIGHT_DIMS[name] if moe_mlp else WEIGHT_DIMS[name]
    sharding = dict(cfg.wt_sharding.get(name, {}))
    if moe_mlp:
        sharding["E"] = cfg.moe.axis
    return LTensor(name, dims, sharding, kind="weight")


def make_input(cfg: StrategyConfig) -> LTensor:
    return LTensor("In", ("B", "T", "D"), cfg.act_sharding, kind="activation")


def forward_steps(cfg: StrategyConfig) -> list[Step]:
    if cfg.pipeline is not None:
        return pipeline_steps(cfg)

    steps: list[Step] = []
    x = make_input(cfg)
    for layer in range(1, cfg.n_layers + 1):
        steps += attention_steps(cfg, x, layer)
        x = steps[-1].out  # type: ignore[union-attr]
        last = layer == cfg.n_layers
        if cfg.moe is not None:
            steps += moe_steps(cfg, x, layer, out_name="Out" if last else "X")
        else:
            steps += mlp_steps(cfg, x, layer, out_name="Out" if last else "X")
        x = steps[-1].out  # type: ignore[union-attr]
    return steps


def attention_steps(cfg: StrategyConfig, x: LTensor, layer: int) -> list[Step]:
    steps: list[Step] = []
    w_qkv = make_weight(cfg, "W_qkv")
    w_o = make_weight(cfg, "W_o")

    qkv_plan, qkv = plan_matmul(
        x, w_qkv, "QKV", "D", layer=layer, phase="attn",
        prefer_reduce_scatter=cfg.prefer_reduce_scatter,
    )
    steps += qkv_plan

    q = replace(qkv, name="Q")
    k = replace(qkv, name="K", kind="kv")
    v = replace(qkv, name="V", kind="kv")
    steps.append(SplitQKVStep(src=qkv, q=q, k=k, v=v, layer=layer, phase="attn"))

    # Context parallelism: attention needs every key/value token, so gather
    # K and V along the axis sharding the sequence dim. Q stays sharded.
    if cfg.kv_context_axis is not None and k.axis_of("T") == cfg.kv_context_axis:
        for t in (k, v):
            gathered = t.gathered("T")
            steps.append(
                AllGatherStep(
                    src=t, out=gathered, axis=cfg.kv_context_axis, dim="T",
                    layer=layer, phase="attn",
                )
            )
            if t.name == "K":
                k = gathered
            else:
                v = gathered

    attn_out = replace(q, name="A")
    steps.append(AttentionCoreStep(q=q, k=k, v=v, out=attn_out, layer=layer, phase="attn"))

    proj_plan, _ = plan_matmul(
        attn_out, w_o, "X", "H", layer=layer, phase="attn",
        scatter_dim="D", prefer_reduce_scatter=cfg.prefer_reduce_scatter,
    )
    steps += proj_plan
    return steps


def mlp_steps(cfg: StrategyConfig, x: LTensor, layer: int, out_name: str = "Out") -> list[Step]:
    steps: list[Step] = []
    w_in = make_weight(cfg, "W_in")
    w_out = make_weight(cfg, "W_out")

    up_plan, tmp = plan_matmul(
        x, w_in, "Tmp", "D", layer=layer, phase="mlp",
        prefer_reduce_scatter=cfg.prefer_reduce_scatter,
    )
    steps += up_plan

    act = replace(tmp, name="Tmp")
    steps.append(GeluStep(src=tmp, out=act, layer=layer, phase="mlp"))

    down_plan, _ = plan_matmul(
        act, w_out, out_name, "F", layer=layer, phase="mlp",
        scatter_dim="D", prefer_reduce_scatter=cfg.prefer_reduce_scatter,
    )
    steps += down_plan
    return steps


def moe_steps(cfg: StrategyConfig, x: LTensor, layer: int, out_name: str = "Out") -> list[Step]:
    """Routed MoE MLP: route -> AllToAll dispatch -> expert MLP -> AllToAll combine."""
    assert cfg.moe is not None
    axis = cfg.moe.axis
    steps: list[Step] = []

    steps.append(RouteStep(tokens=x, n_experts=cfg.moe.n_experts, layer=layer, phase="moe"))

    # Dispatch: each device ends up holding the S tokens routed to ITS expert.
    routed = LTensor("X", ("E", "S", "D"), {"E": axis}, kind="activation")
    steps.append(
        AllToAllStep(src=x, out=routed, axis=axis, direction="dispatch", layer=layer, phase="moe")
    )

    w_in = make_weight(cfg, "W_in")
    w_out = make_weight(cfg, "W_out")

    # Expert compute is local: each device multiplies its token group by its
    # expert's weights (batched over the co-sharded E dim, so no comms).
    tmp = LTensor("Tmp", ("E", "S", "F"), {"E": axis}, kind="activation")
    steps.append(MatMulStep(a=routed, b=w_in, out=tmp, contract="D", layer=layer, phase="moe"))
    steps.append(GeluStep(src=tmp, out=tmp, layer=layer, phase="moe"))
    expert_out = LTensor("X", ("E", "S", "D"), {"E": axis}, kind="activation")
    steps.append(
        MatMulStep(a=tmp, b=w_out, out=expert_out, contract="F", layer=layer, phase="moe")
    )

    # Combine: tokens fly home to their original devices/positions.
    out = LTensor(out_name, ("B", "T", "D"), cfg.act_sharding, kind="activation")
    steps.append(
        AllToAllStep(
            src=expert_out, out=out, axis=axis, direction="combine", layer=layer, phase="moe"
        )
    )
    return steps


def pipeline_steps(cfg: StrategyConfig) -> list[Step]:
    """GPipe-style forward schedule: stage s runs microbatch m at tick m + s.

    Each stage holds one layer (n_stages == n_layers for our 2-layer model).
    Events are ordered by (tick, stage); the Gantt inset renders straight off
    the tick fields, so schedule and animation cannot drift apart.
    """
    assert cfg.pipeline is not None
    pp = cfg.pipeline
    acts = LTensor("In", ("B", "T", "D"), cfg.act_sharding, kind="activation")

    events: list[Step] = []
    for m in range(pp.n_microbatches):
        for s in range(pp.n_stages):
            tick = m + s
            events.append(
                StageComputeStep(
                    stage=s, tick=tick, layer=s + 1, phase="pipeline", microbatch=m
                )
            )
            if s < pp.n_stages - 1:
                events.append(
                    P2PSendStep(
                        tensor=acts, src_stage=s, dst_stage=s + 1, tick=tick,
                        layer=s + 1, phase="pipeline", microbatch=m,
                    )
                )
    events.sort(key=lambda e: (e.tick, e.stage if isinstance(e, StageComputeStep) else e.src_stage + 0.5))  # type: ignore[attr-defined]
    return events


def count_collectives(steps: list[Step]) -> dict[str, int]:
    """Per-op collective counts — used by tests and by scene summary cards."""
    from tpviz.core.steps import CollectiveStep

    counts: dict[str, int] = {}
    for s in steps:
        if isinstance(s, CollectiveStep):
            counts[s.op] = counts.get(s.op, 0) + 1
        elif isinstance(s, P2PSendStep):
            counts["P2P"] = counts.get("P2P", 0) + 1
    return counts
