"""Training-step generation: forward (with activation saving) + backward.

The backward pass is derived with the SAME sharded-matmul engine as the
forward: for every forward matmul Y = X · W we emit
    dX = dY · Wᵀ          (contract the output dim)
    dW = Xᵀ · dY          (contract B and T — the multi-dim contraction)
and the engine's four cases produce each strategy's collectives:
  DP:   dW partial over B_X  -> AllReduce_X (the classic gradient all-reduce)
  FSDP: Wᵀ re-gathered jit for dX; dW partial -> ReduceScatter onto the shard dim
  TP:   mirrors forward — AllGather(dY) in, ReduceScatter(dX) out; dW local
  CP:   dK/dV partial over the context axis -> ReduceScatter over T;
        dW partial over T_X -> AllReduce
  EP:   AllToAll of token gradients (dispatch/combine), expert dW local

Saved activations are exactly each forward matmul's (possibly gathered) input
operand, plus Q/K/V for the attention core — "save what the matmul consumed".
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from tpviz.core.engine import plan_matmul
from tpviz.core.mesh import StrategyConfig
from tpviz.core.model import forward_steps, make_weight
from tpviz.core.steps import (
    AllGatherStep,
    AllToAllStep,
    AttentionBwdStep,
    AttentionCoreStep,
    GeluStep,
    GradInitStep,
    MatMulStep,
    MergeQKVStep,
    P2PSendStep,
    ReduceScatterStep,
    SaveActivationStep,
    StageComputeStep,
    Step,
)
from tpviz.core.tensors import LTensor


@dataclass
class LayerRecord:
    """What the forward pass of one layer saved (post-gather operand forms)."""

    attn_in: LTensor | None = None
    q: LTensor | None = None
    k: LTensor | None = None
    v: LTensor | None = None
    attn_a: LTensor | None = None  # input to the W_o matmul
    mlp_in: LTensor | None = None  # input to the W_in matmul (MoE: routed tokens)
    tmp: LTensor | None = None  # input to the W_out matmul
    out: LTensor | None = None  # the layer's block output


def annotate_forward(cfg: StrategyConfig) -> tuple[list[Step], dict[int, LayerRecord]]:
    """Forward steps with SaveActivationSteps injected; per-layer records."""
    steps: list[Step] = []
    recs: dict[int, LayerRecord] = {la: LayerRecord() for la in range(1, cfg.n_layers + 1)}
    slot_count: dict[tuple[int, str], int] = {}

    def save(t: LTensor, layer: int, phase: str) -> SaveActivationStep:
        key = (layer, phase)
        slot = slot_count.get(key, 0)
        slot_count[key] = slot + 1
        return SaveActivationStep(t=t, slot=slot, layer=layer, phase=phase)

    for step in forward_steps(cfg):
        steps.append(step)
        rec = recs.get(step.layer)
        if isinstance(step, MatMulStep) and step.b.kind == "weight":
            steps.append(save(step.a, step.layer, step.phase))
            if step.b.name == "W_qkv":
                rec.attn_in = step.a
            elif step.b.name == "W_o":
                rec.attn_a = step.a
            elif step.b.name == "W_in":
                rec.mlp_in = step.a
            elif step.b.name == "W_out":
                rec.tmp = step.a
        elif isinstance(step, AttentionCoreStep):
            for t in (step.q, step.k, step.v):
                steps.append(save(t, step.layer, step.phase))
            rec.q, rec.k, rec.v = step.q, step.k, step.v
        if hasattr(step, "out") and getattr(step.out, "kind", None) == "activation":
            rec.out = step.out
    return steps, recs


def _gathered_form(steps: list[Step], t: LTensor) -> LTensor:
    """The post-AllGather form of `t` if one of `steps` gathered it."""
    for s in steps:
        if isinstance(s, AllGatherStep) and s.src.name == t.name and s.src.kind == t.kind:
            t = s.out
    return t


def _wt_scatter_dim(cfg: StrategyConfig, name: str) -> str | None:
    sharding = cfg.wt_sharding.get(name, {})
    return next(iter(sharding), None)


def backward_mlp(cfg: StrategyConfig, rec: LayerRecord, d_out: LTensor, layer: int
                 ) -> tuple[list[Step], LTensor]:
    """dOut -> dX through the MLP, plus dW_in / dW_out."""
    phase = "mlp"
    steps: list[Step] = []
    w_out, w_in = make_weight(cfg, "W_out"), make_weight(cfg, "W_in")

    s, dtmp = plan_matmul(
        d_out, w_out.transposed(), "dTmp", "D", layer=layer, phase=phase, out_kind="grad",
        prefer_reduce_scatter=cfg.prefer_reduce_scatter,
    )
    steps += s
    d_out_used = _gathered_form(s, d_out)

    s, _ = plan_matmul(
        rec.tmp, d_out_used, "dW_out", ("B", "T"), layer=layer, phase=phase,
        scatter_dim=_wt_scatter_dim(cfg, "W_out"), out_kind="grad",
        prefer_reduce_scatter=cfg.prefer_reduce_scatter,
    )
    steps += s

    steps.append(GeluStep(src=dtmp, out=dtmp, layer=layer, phase=phase,
                          caption="through the gelu: dTmp ⊙ gelu′"))

    s, dx = plan_matmul(
        dtmp, w_in.transposed(), "dX", "F", layer=layer, phase=phase,
        scatter_dim="D", out_kind="grad", prefer_reduce_scatter=cfg.prefer_reduce_scatter,
    )
    steps += s
    dtmp_used = _gathered_form(s, dtmp)

    s, _ = plan_matmul(
        rec.mlp_in, dtmp_used, "dW_in", ("B", "T"), layer=layer, phase=phase,
        scatter_dim=_wt_scatter_dim(cfg, "W_in"), out_kind="grad",
        prefer_reduce_scatter=cfg.prefer_reduce_scatter,
    )
    steps += s
    for st in steps:
        st.backward = True
    return steps, dx


def backward_moe(cfg: StrategyConfig, rec: LayerRecord, d_out: LTensor, layer: int
                 ) -> tuple[list[Step], LTensor]:
    """Token gradients AllToAll to their experts, expert backward, AllToAll home."""
    assert cfg.moe is not None
    phase, axis = "moe", cfg.moe.axis
    steps: list[Step] = []
    w_in, w_out = make_weight(cfg, "W_in"), make_weight(cfg, "W_out")

    dxe = LTensor("dX", ("E", "S", "D"), {"E": axis}, kind="grad")
    steps.append(AllToAllStep(src=d_out, out=dxe, axis=axis, direction="dispatch",
                              layer=layer, phase=phase))

    dtmp = LTensor("dTmp", ("E", "S", "F"), {"E": axis}, kind="grad")
    steps.append(MatMulStep(a=dxe, b=w_out.transposed(), out=dtmp, contract="D",
                            layer=layer, phase=phase))
    dw_out = LTensor("dW_out", ("E", "F", "D"), {"E": axis}, kind="grad")
    steps.append(MatMulStep(a=rec.tmp, b=dxe, out=dw_out, contract="S",
                            layer=layer, phase=phase))
    steps.append(GeluStep(src=dtmp, out=dtmp, layer=layer, phase=phase,
                          caption="through the gelu: dTmp ⊙ gelu′"))
    dtok = LTensor("dX", ("E", "S", "D"), {"E": axis}, kind="grad")
    steps.append(MatMulStep(a=dtmp, b=w_in.transposed(), out=dtok, contract="F",
                            layer=layer, phase=phase))
    dw_in = LTensor("dW_in", ("E", "D", "F"), {"E": axis}, kind="grad")
    steps.append(MatMulStep(a=rec.mlp_in, b=dtmp, out=dw_in, contract="S",
                            layer=layer, phase=phase))

    dx = LTensor("dX", ("B", "T", "D"), cfg.act_sharding, kind="grad")
    steps.append(AllToAllStep(src=dtok, out=dx, axis=axis, direction="combine",
                              layer=layer, phase=phase))
    for st in steps:
        st.backward = True
    return steps, dx


def backward_attention(cfg: StrategyConfig, rec: LayerRecord, d_out: LTensor, layer: int
                       ) -> tuple[list[Step], LTensor]:
    """dX (block-output grad) -> d(block input), plus dW_qkv / dW_o and the
    attention-core backward (with CP's dK/dV ReduceScatter)."""
    phase = "attn"
    steps: list[Step] = []
    w_o, w_qkv = make_weight(cfg, "W_o"), make_weight(cfg, "W_qkv")

    s, da = plan_matmul(
        d_out, w_o.transposed(), "dA", "D", layer=layer, phase=phase, out_kind="grad",
        prefer_reduce_scatter=cfg.prefer_reduce_scatter,
    )
    steps += s
    d_out_used = _gathered_form(s, d_out)

    s, _ = plan_matmul(
        rec.attn_a, d_out_used, "dW_o", ("B", "T"), layer=layer, phase=phase,
        scatter_dim=_wt_scatter_dim(cfg, "W_o"), out_kind="grad",
        prefer_reduce_scatter=cfg.prefer_reduce_scatter,
    )
    steps += s

    # attention core backward: dA -> dQ, dK, dV (uses the saved Q, K, V)
    dq = replace(rec.q.grad(), name="dQ")
    ctx = cfg.kv_context_axis
    cp_active = ctx is not None and rec.q.axis_of("T") == ctx
    if cp_active:
        # every query shard contributes to every dK/dV token: partial sums
        dk = replace(rec.k.grad(), name="dK", partial=frozenset({ctx}))
        dv = replace(rec.v.grad(), name="dV", partial=frozenset({ctx}))
        steps.append(AttentionBwdStep(da=da, dq=dq, dk=dk, dv=dv, layer=layer, phase=phase))
        dk_res, dv_res = dk.scattered("T", ctx), dv.scattered("T", ctx)
        steps.append(ReduceScatterStep(src=dk, out=dk_res, axis=ctx, scatter_dim="T",
                                       layer=layer, phase=phase))
        steps.append(ReduceScatterStep(src=dv, out=dv_res, axis=ctx, scatter_dim="T",
                                       layer=layer, phase=phase))
        dk, dv = dk_res, dv_res
    else:
        dk = replace(rec.k.grad(), name="dK")
        dv = replace(rec.v.grad(), name="dV")
        steps.append(AttentionBwdStep(da=da, dq=dq, dk=dk, dv=dv, layer=layer, phase=phase))

    dqkv = replace(dq, name="dQKV")
    steps.append(MergeQKVStep(q=dq, k=dk, v=dv, out=dqkv, layer=layer, phase=phase))

    s, dx = plan_matmul(
        dqkv, w_qkv.transposed(), "dX", "H", layer=layer, phase=phase,
        scatter_dim="D", out_kind="grad", prefer_reduce_scatter=cfg.prefer_reduce_scatter,
    )
    steps += s
    dqkv_used = _gathered_form(s, dqkv)

    s, _ = plan_matmul(
        rec.attn_in, dqkv_used, "dW_qkv", ("B", "T"), layer=layer, phase=phase,
        scatter_dim=_wt_scatter_dim(cfg, "W_qkv"), out_kind="grad",
        prefer_reduce_scatter=cfg.prefer_reduce_scatter,
    )
    steps += s
    for st in steps:
        st.backward = True
    return steps, dx


def train_steps(cfg: StrategyConfig) -> list[Step]:
    """Forward (saving activations) + backward, as one step list."""
    if cfg.pipeline is not None:
        return pipeline_train_steps(cfg)

    steps, recs = annotate_forward(cfg)
    final_out = recs[cfg.n_layers].out
    d_out = replace(final_out.grad(), name="dOut")
    steps.append(
        GradInitStep(
            out=d_out, layer=cfg.n_layers, phase="mlp" if cfg.moe is None else "moe",
            backward=True,
            caption="Backward starts from the loss: dOut = ∂L/∂Out",
        )
    )
    for layer in range(cfg.n_layers, 0, -1):
        rec = recs[layer]
        if cfg.moe is not None:
            s, dx = backward_moe(cfg, rec, d_out, layer)
        else:
            s, dx = backward_mlp(cfg, rec, d_out, layer)
        steps += s
        s, d_out = backward_attention(cfg, rec, dx, layer)
        steps += s
    return steps


def pipeline_train_steps(cfg: StrategyConfig) -> list[Step]:
    """GPipe forward then backward. Backward compute cells are 2 ticks wide —
    the backward pass costs ~2x the forward."""
    assert cfg.pipeline is not None
    pp = cfg.pipeline
    S, M = pp.n_stages, pp.n_microbatches
    acts = LTensor("In", ("B", "T", "D"), cfg.act_sharding, kind="activation")
    grads = LTensor("dX", ("B", "T", "D"), cfg.act_sharding, kind="grad")

    events: list[Step] = list(forward_steps(cfg))
    f_end = M + S - 1  # forward occupies ticks [0, f_end)

    for m in range(M - 1, -1, -1):
        for s in range(S - 1, -1, -1):
            tick = f_end + 2 * (M - 1 - m) + 2 * (S - 1 - s)
            events.append(
                StageComputeStep(stage=s, tick=tick, layer=s + 1, phase="pipeline",
                                 microbatch=m, backward=True)
            )
            if s > 0:
                events.append(
                    P2PSendStep(tensor=grads, src_stage=s, dst_stage=s - 1, tick=tick + 1,
                                layer=s + 1, phase="pipeline", microbatch=m, backward=True)
                )
    return events


def count_matmuls(steps: list[Step]) -> dict[str, int]:
    counts = {"forward": 0, "backward": 0}
    for s in steps:
        if isinstance(s, MatMulStep):
            counts["backward" if s.backward else "forward"] += 1
    return counts
