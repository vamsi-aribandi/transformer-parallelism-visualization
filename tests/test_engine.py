"""Pin every strategy's collective sequence to what the scaling book prescribes."""

from tpviz import configs, notation
from tpviz.core.model import count_collectives, forward_steps
from tpviz.core.steps import (
    AllGatherStep,
    AllToAllStep,
    CollectiveStep,
    MatMulStep,
    P2PSendStep,
    ReduceScatterStep,
    StageComputeStep,
)
from tpviz.core.tensors import LTensor


def collectives(steps):
    return [s for s in steps if isinstance(s, CollectiveStep)]


def test_dp_forward_has_zero_communication():
    steps = forward_steps(configs.DP)
    assert collectives(steps) == []
    assert count_collectives(steps) == {}
    out = [s for s in steps if isinstance(s, MatMulStep)][-1].out
    assert out.name == "Out" and out.sharding == {"B": "X"}


def test_fsdp_gathers_each_weight_just_in_time():
    steps = forward_steps(configs.FSDP)
    ags = collectives(steps)
    assert all(isinstance(s, AllGatherStep) for s in ags)
    # 4 weights per layer x 2 layers, every gather is a jit weight gather on X
    assert len(ags) == 8
    assert all(s.jit and s.axis == "X" and s.src.kind == "weight" for s in ags)
    assert [s.src.name for s in ags if s.layer == 1] == ["W_qkv", "W_o", "W_in", "W_out"]
    # activations stay batch-sharded throughout
    out = [s for s in steps if isinstance(s, MatMulStep)][-1].out
    assert out.sharding == {"B": "X"}


def test_tp_allgather_in_reducescatter_out_per_block():
    steps = forward_steps(configs.TP)
    layer1 = [s for s in collectives(steps) if s.layer == 1]
    kinds = [(type(s).__name__, s.phase) for s in layer1]
    assert kinds == [
        ("AllGatherStep", "attn"),
        ("ReduceScatterStep", "attn"),
        ("AllGatherStep", "mlp"),
        ("ReduceScatterStep", "mlp"),
    ]
    ag_attn, rs_attn, ag_mlp, rs_mlp = layer1
    # AllGather brings in the activations (not weights), along Y over D
    assert ag_attn.src.name == "In" and ag_attn.dim == "D" and ag_attn.axis == "Y"
    assert not ag_attn.jit
    # ReduceScatter resolves the {U_Y} partial onto D
    assert rs_attn.src.partial == frozenset({"Y"})
    assert rs_attn.scatter_dim == "D" and rs_attn.out.sharding == {"D": "Y"}
    assert rs_mlp.out.sharding == {"D": "Y"}
    assert count_collectives(steps) == {"AllGather": 4, "ReduceScatter": 4}


def test_cp_gathers_kv_only_in_attention():
    steps = forward_steps(configs.CP)
    cs = collectives(steps)
    assert all(isinstance(s, AllGatherStep) for s in cs)
    assert [(s.src.name, s.dim, s.axis, s.phase) for s in cs if s.layer == 1] == [
        ("K", "T", "X", "attn"),
        ("V", "T", "X", "attn"),
    ]
    assert all(s.phase == "attn" for s in cs)  # the MLP never communicates
    assert count_collectives(steps) == {"AllGather": 4}
    out = [s for s in steps if isinstance(s, MatMulStep)][-1].out
    assert out.sharding == {"T": "X"}


def test_ep_dispatch_and_combine_alltoall_per_moe_layer():
    steps = forward_steps(configs.EP)
    a2a = [s for s in steps if isinstance(s, AllToAllStep)]
    assert [(s.direction, s.layer) for s in a2a] == [
        ("dispatch", 1), ("combine", 1), ("dispatch", 2), ("combine", 2),
    ]
    assert all(s.axis == "Z" for s in a2a)
    # attention itself is communication-free (it rides on the batch sharding)
    assert all(s.phase == "moe" for s in collectives(steps))
    assert count_collectives(steps) == {"AllToAll": 4}


def test_pp_gpipe_schedule_shape():
    steps = forward_steps(configs.PP)
    computes = [s for s in steps if isinstance(s, StageComputeStep)]
    sends = [s for s in steps if isinstance(s, P2PSendStep)]
    assert len(computes) == 8  # 2 stages x 4 microbatches
    assert len(sends) == 4  # one send per microbatch between the 2 stages
    # GPipe: stage s runs microbatch m at tick m + s
    assert all(c.tick == c.microbatch + c.stage for c in computes)
    # overlap exists: at tick 1, stage 0 runs mb1 while stage 1 runs mb0
    tick1 = [(c.stage, c.microbatch) for c in computes if c.tick == 1]
    assert set(tick1) == {(0, 1), (1, 0)}


def test_notation_matches_the_book():
    t = LTensor("In", ("B", "T", "D"), {"B": "X"})
    assert t.tex() == r"\mathrm{In}[B_{X},\, T,\, D]"
    partial = LTensor("Out", ("B", "T", "D"), {}, partial=frozenset({"Y"}))
    assert partial.tex() == r"\mathrm{Out}[B,\, T,\, D]\{U_{Y}\}"
    assert notation.collective_tex("AllGather", "X", t, t.gathered("B")) == (
        r"\text{AllGather}_{X}\; \mathrm{In}[B_{X},\, T,\, D] \,\to\, \mathrm{In}[B,\, T,\, D]"
    )


def test_engine_rejects_contract_sharded_on_two_axes():
    import pytest

    from tpviz.core.engine import plan_matmul

    a = LTensor("A", ("B", "D"), {"D": "X"})
    b = LTensor("W", ("D", "F"), {"D": "Y"}, kind="weight")
    with pytest.raises(ValueError):
        plan_matmul(a, b, "O", "D", layer=1, phase="mlp")
