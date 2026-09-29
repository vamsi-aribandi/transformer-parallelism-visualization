"""Symbolic sharded-matmul planner — the scaling book's four cases.

Given two logical tensors and a contracting dim, emit the collectives the
multiplication requires and the sharding of the result. This is what makes
multi-axis combinations (up to 5D) config-driven: the same rules that produce
FSDP's just-in-time weight AllGather also produce TP's activation AllGather
and TP's partial-sum ReduceScatter, with no per-strategy branching.

Book cases (https://jax-ml.github.io/scaling-book/sharding/), applied PER
MESH-AXIS LETTER so that a dim sharded over several axes at once (`B_{XZ}`)
composes:
  1. Contracting dim unsharded on both operands -> local matmul, no comms.
  2. Contracting dim sharded on one operand only (along some axis) -> AllGather
     that operand along that axis, then case 1. When the two operands shard the
     contracting dim along DIFFERENT axes (FSDP x TP: In[.., D_Y] . W[D_X, ..])
     both get gathered, each along its own axis — the weight first (jit).
  3. Contracting dim sharded on both via the SAME axis -> local matmul yields a
     partial sum {U_axis}; resolve with ReduceScatter (onto scatter_dim) or AllReduce.
  4. The same axis shards DIFFERENT non-contracting dims of the two operands
     (output would be double-sharded on one axis) -> AllGather one operand
     first. We gather the weight, preserving the activation's sharding (this is
     FSDP's second matmul). The same axis sharding the SAME dim on both (a
     batched dim like E_Z on routed tokens and expert weights) is fine.
"""

from __future__ import annotations

from tpviz.core.mesh import AXIS_ORDER
from tpviz.core.steps import (
    AllGatherStep,
    AllReduceStep,
    MatMulStep,
    ReduceScatterStep,
    Step,
)
from tpviz.core.tensors import LTensor


def _letter_dims(t: LTensor, contracts: tuple[str, ...]) -> dict[str, str]:
    """mesh-axis letter -> the NON-contracting dim it shards on `t`."""
    return {
        a: dim for dim, axes in t.sharding.items() if dim not in contracts for a in axes
    }


def plan_matmul(
    a: LTensor,
    b: LTensor,
    out_name: str,
    contract: str | tuple[str, ...],
    *,
    layer: int,
    phase: str,
    scatter_dim: str | None = None,
    scatter_axis: str | None = None,
    prefer_reduce_scatter: bool = True,
    out_kind: str = "activation",
) -> tuple[list[Step], LTensor]:
    """Plan a · b contracting over `contract` (one dim, or several — weight
    gradients contract over both B and T). Returns (steps, result tensor).

    `scatter_dim` / `scatter_axis`: when the product is a partial sum, the
    partial over `scatter_axis` (default: the first partial axis) resolves with
    a ReduceScatter onto `scatter_dim`; any other partial axes AllReduce."""
    contracts = (contract,) if isinstance(contract, str) else tuple(contract)
    for c in contracts:
        if c not in a.dims or c not in b.dims:
            raise ValueError(f"contract dim {c!r} must be in both {a.dims} and {b.dims}")
    if a.partial or b.partial:
        raise ValueError("operands with unresolved partial sums cannot be multiplied")

    steps: list[Step] = []
    partial_axes: list[str] = []

    def gather(t: LTensor, dim: str, axis: str) -> LTensor:
        gathered = t.gathered(dim, axis)
        steps.append(
            AllGatherStep(
                src=t, out=gathered, axis=axis, dim=dim, jit=t.kind == "weight",
                layer=layer, phase=phase,
            )
        )
        return gathered

    for c in contracts:
        a_ax, b_ax = a.axes_of(c), b.axes_of(c)
        # Case 3: letters shared by both operands -> partial sums
        partial_axes += [x for x in a_ax if x in b_ax]
        # Case 2: letters only one operand has -> gather it along that axis.
        # Weights first: FSDP's jit gather precedes TP's activation gather.
        first, second = (b, a) if b.kind == "weight" or a.kind != "weight" else (a, b)
        for t_is_b in (first is b, second is b):
            t, other = (b, a) if t_is_b else (a, b)
            for x in t.axes_of(c):
                if x not in other.axes_of(c):
                    t = gather(t, c, x)
            if t_is_b:
                b = t
            else:
                a = t

    # Case 4: one letter would shard two different output dims -> AllGather
    # the weight (or `b` if neither operand is a weight) along it.
    la, lb = _letter_dims(a, contracts), _letter_dims(b, contracts)
    conflict = [x for x in la if x in lb and la[x] != lb[x]]
    if conflict:
        victim_is_b = b.kind == "weight" or a.kind != "weight"
        victim = b if victim_is_b else a
        for x in sorted(conflict, key=lambda x: AXIS_ORDER.find(x) % 99):
            dim = (lb if victim_is_b else la)[x]
            victim = gather(victim, dim, x)
        if victim_is_b:
            b = victim
        else:
            a = victim

    # output dims: a's non-contracting dims, then b's not already present
    # (a batched dim like E appears once)
    out_dims = tuple(d for d in a.dims if d not in contracts)
    out_dims += tuple(d for d in b.dims if d not in contracts and d not in out_dims)
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

    ordered = sorted(dict.fromkeys(partial_axes), key=lambda x: AXIS_ORDER.find(x) % 99)
    if scatter_axis is None or scatter_axis not in ordered:
        scatter_axis = ordered[0] if ordered else None
    for axis in ordered:
        if scatter_dim and prefer_reduce_scatter and axis == scatter_axis:
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
        else:
            resolved = out.reduced(axis)
            steps.append(
                AllReduceStep(src=out, out=resolved, axis=axis, layer=layer, phase=phase)
            )
        out = resolved

    return steps, out
