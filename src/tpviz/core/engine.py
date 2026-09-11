"""Symbolic sharded-matmul planner — the scaling book's four cases.

Given two logical tensors and a contracting dim, emit the collectives the
multiplication requires and the sharding of the result. This is what makes
multi-axis combinations (up to 5D) config-driven: the same rules that produce
FSDP's just-in-time weight AllGather also produce TP's activation AllGather
and TP's partial-sum ReduceScatter, with no per-strategy branching.

Book cases (https://jax-ml.github.io/scaling-book/sharding/):
  1. Contracting dim unsharded on both operands -> local matmul, no comms.
  2. Contracting dim sharded on exactly one operand -> AllGather that operand
     along its axis, then case 1.
  3. Contracting dim sharded on both via the SAME axis -> local matmul yields a
     partial sum {U_axis}; resolve with ReduceScatter (onto scatter_dim) or AllReduce.
  4. The same axis shards NON-contracting dims of both operands (output would be
     double-sharded on one axis) -> AllGather one operand first. We gather the
     weight, preserving the activation's sharding (this is FSDP's second matmul).
"""

from __future__ import annotations

from tpviz.core.steps import (
    AllGatherStep,
    AllReduceStep,
    MatMulStep,
    ReduceScatterStep,
    Step,
)
from tpviz.core.tensors import LTensor


def plan_matmul(
    a: LTensor,
    b: LTensor,
    out_name: str,
    contract: str | tuple[str, ...],
    *,
    layer: int,
    phase: str,
    scatter_dim: str | None = None,
    prefer_reduce_scatter: bool = True,
    out_kind: str = "activation",
) -> tuple[list[Step], LTensor]:
    """Plan a · b contracting over `contract` (one dim, or several — weight
    gradients contract over both B and T). Returns (steps, result tensor)."""
    contracts = (contract,) if isinstance(contract, str) else tuple(contract)
    for c in contracts:
        if c not in a.dims or c not in b.dims:
            raise ValueError(f"contract dim {c!r} must be in both {a.dims} and {b.dims}")
    if a.partial or b.partial:
        raise ValueError("operands with unresolved partial sums cannot be multiplied")

    steps: list[Step] = []
    partial_axes: list[str] = []
    for c in contracts:
        a_ax, b_ax = a.axis_of(c), b.axis_of(c)
        if a_ax and b_ax and a_ax != b_ax:
            raise ValueError(
                f"contract dim {c!r} sharded over different axes ({a_ax} vs {b_ax})"
            )
        if a_ax and b_ax:
            # Case 3: local matmul over shards produces an unreduced partial sum.
            partial_axes.append(a_ax)
        elif a_ax or b_ax:
            # Case 2: AllGather the operand sharded along the contracting dim.
            victim = a if a_ax else b
            gathered = victim.gathered(c)
            steps.append(
                AllGatherStep(
                    src=victim,
                    out=gathered,
                    axis=a_ax or b_ax,  # the one that is set
                    dim=c,
                    jit=victim.kind == "weight",
                    layer=layer,
                    phase=phase,
                )
            )
            if a_ax:
                a = gathered
            else:
                b = gathered

    # Case 4: same axis would shard two output dims -> AllGather the weight
    # (or `b` if neither operand is a weight) along its conflicting dims.
    a_out_axes = {ax for d, ax in a.sharding.items() if d not in contracts}
    b_out_axes = {ax for d, ax in b.sharding.items() if d not in contracts}
    conflict = a_out_axes & b_out_axes
    if conflict:
        victim_is_b = b.kind == "weight" or a.kind != "weight"
        victim = b if victim_is_b else a
        for dim in list(victim.dims):
            ax = victim.axis_of(dim)
            if dim not in contracts and ax in conflict:
                gathered = victim.gathered(dim)
                steps.append(
                    AllGatherStep(
                        src=victim,
                        out=gathered,
                        axis=ax,
                        dim=dim,
                        jit=victim.kind == "weight",
                        layer=layer,
                        phase=phase,
                    )
                )
                victim = gathered
        if victim_is_b:
            b = victim
        else:
            a = victim

    out_dims = tuple(d for d in a.dims if d not in contracts) + tuple(
        d for d in b.dims if d not in contracts
    )
    out_sharding = {d: ax for d, ax in a.sharding.items() if d not in contracts}
    out_sharding |= {d: ax for d, ax in b.sharding.items() if d not in contracts}
    out = LTensor(
        out_name,
        out_dims,
        out_sharding,
        partial=frozenset(partial_axes),
        kind=out_kind,
    )
    steps.append(
        MatMulStep(a=a, b=b, out=out, contract=",".join(contracts), layer=layer, phase=phase)
    )

    for axis in partial_axes:
        if scatter_dim and prefer_reduce_scatter:
            resolved = out.scattered(scatter_dim, axis)
            steps.append(
                ReduceScatterStep(
                    src=out,
                    out=resolved,
                    axis=axis,
                    scatter_dim=scatter_dim,
                    layer=layer,
                    phase=phase,
                )
            )
            scatter_dim = None  # only the first resolution can scatter a new dim
        else:
            resolved = out.reduced(axis)
            steps.append(
                AllReduceStep(src=out, out=resolved, axis=axis, layer=layer, phase=phase)
            )
        out = resolved

    return steps, out
