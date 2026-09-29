"""The 5D combination: every collective attributed to exactly one mesh axis.

FSDP (X) x TP (Y) x CP (C) x EP (Z) x PP (stage), two of each. These tests
pin the derived forward AND backward collective sequences per block."""

from tpviz import configs
from tpviz.core.backward import count_matmuls, train_steps
from tpviz.core.mesh import Mesh, canon_axes
from tpviz.core.model import count_collectives, forward_steps
from tpviz.core.steps import (
    AllGatherStep,
    AllReduceStep,
    AllToAllStep,
    CollectiveStep,
    MatMulStep,
    P2PSendStep,
    ReduceScatterStep,
    SaveActivationStep,
)
from tpviz.core.tensors import LTensor

CFG = configs.FIVE_D


def sig(s):
    """(op, axis, tensor) — the attribution a reader sees for a collective."""
    return (type(s).__name__.removesuffix("Step"), s.axis, s.src.name)


def colls(steps, *, layer=None, phase=None, backward=None):
    out = []
    for s in steps:
        if not isinstance(s, (CollectiveStep, P2PSendStep)):
            continue
        if layer is not None and s.layer != layer:
            continue
        if phase is not None and s.phase != phase:
            continue
        if backward is not None and s.backward != backward:
            continue
        out.append(s)
    return out


def test_multi_axis_notation():
    t = LTensor("In", ("B", "T", "D"), {"B": "ZX", "T": "C", "D": "Y"})
    assert t.sharding == {"B": "XZ", "T": "C", "D": "Y"}  # canonical letter order
    assert t.tex() == r"\mathrm{In}[B_{XZ},\, T_{C},\, D_{Y}]"
    p = LTensor("dW", ("D", "H"), {"H": "Y"}, partial=frozenset({"C", "X", "Z"}), kind="grad")
    assert p.tex().endswith(r"\{U_{XZC}\}")
    assert p.scattered("D", "X").tex().endswith(r"[D_{X},\, H_{Y}]\{U_{ZC}\}")
    assert t.gathered("B", "Z").sharding == {"B": "X", "T": "C", "D": "Y"}
    assert canon_axes("CXZ") == "XZC"


def test_mesh_compound_coordinates():
    m = CFG.mesh
    assert m.n_devices == 32
    assert m.size("XZ") == 4 and m.size("stage") == 2
    d = m.index(X=1, Y=0, C=1, Z=0, stage=1)
    assert m.coords(d) == {"X": 1, "Y": 0, "C": 1, "Z": 0, "stage": 1}
    assert m.coord("XZ", d) == 2  # row-major over (X, Z): 1*2 + 0
    assert m.coord("C", d) == 1


def test_activation_layout_and_weights():
    steps = forward_steps(CFG)
    first = next(s for s in steps if isinstance(s, MatMulStep))
    # the input was gathered over the tensor axis for the first matmul
    assert first.a.sharding == {"B": "XZ", "T": "C"}
    assert first.b.sharding == {"H": "Y"}  # W_qkv after its FSDP gather
    assert first.out.sharding == {"B": "XZ", "T": "C", "H": "Y"}


def test_forward_attention_block_collectives():
    steps = forward_steps(CFG)
    assert [sig(s) for s in colls(steps, layer=1, phase="attn")] == [
        ("AllGather", "X", "W_qkv"),  # FSDP: jit weight gather
        ("AllGather", "Y", "In"),     # TP: gather activations in
        ("AllGather", "C", "K"),      # CP: every query sees every key
        ("AllGather", "C", "V"),
        ("AllGather", "X", "W_o"),    # FSDP
        ("ReduceScatter", "Y", "X"),  # TP: partial sum out
    ]


def test_forward_moe_block_collectives_and_routed_sharding():
    steps = forward_steps(CFG)
    moe = colls(steps, layer=1, phase="moe")
    assert [sig(s) for s in moe] == [
        ("AllToAll", "Z", "X"),        # EP dispatch
        ("AllGather", "X", "W_in"),    # FSDP on the expert weights
        ("AllGather", "Y", "X"),       # TP inside the expert
        ("AllGather", "X", "W_out"),
        ("ReduceScatter", "Y", "X"),
        ("AllToAll", "Z", "X"),        # EP combine
    ]
    dispatch = moe[0]
    # the AllToAll re-sorts tokens along Z only: S stays sharded over the
    # other token axes (X from the batch, C from the sequence), D stays on Y
    assert dispatch.out.dims == ("E", "S", "D")
    assert dispatch.out.sharding == {"E": "Z", "S": "XC", "D": "Y"}
    assert moe[-1].out.sharding == CFG.act_sharding


def test_pipeline_sends_between_layers():
    steps = train_steps(CFG)
    sends = [s for s in steps if isinstance(s, P2PSendStep)]
    assert [(p.src_stage, p.dst_stage, p.backward) for p in sends] == [(0, 1, False), (1, 0, True)]
    assert sends[0].tensor.sharding == CFG.act_sharding
    assert sends[1].tensor.name == "dX" and sends[1].note_tex
    # the forward send sits between layer 1's last op and layer 2's first
    i = steps.index(sends[0])
    assert steps[i - 1].layer == 1 and steps[i + 1].layer == 2
    # and the whole forward of layer 1 (stage 0) precedes it
    assert all(s.layer == 1 for s in steps[:i] if not isinstance(s, P2PSendStep))


def test_backward_attention_weight_grads_reduce_over_every_token_axis():
    steps = train_steps(CFG)
    attn = colls(steps, layer=2, phase="attn", backward=True)
    assert [sig(s) for s in attn] == [
        ("AllGather", "X", "W_o"),        # FSDP re-gather
        ("AllGather", "Y", "dX"),         # TP: dual of the forward ReduceScatter
        ("ReduceScatter", "X", "dW_o"),   # FSDP: grad back onto the weight shards
        ("AllReduce", "Z", "dW_o"),       # the expert axis replicates attention
        ("AllReduce", "C", "dW_o"),       # the context axis replicates weights
        ("AllGather", "C", "K"),          # CP: saved K/V shards re-gathered
        ("AllGather", "C", "V"),
        ("ReduceScatter", "C", "dK"),     # CP: dK/dV back over the sequence
        ("ReduceScatter", "C", "dV"),
        ("AllGather", "X", "W_qkv"),
        ("ReduceScatter", "Y", "dX"),     # TP: dual of the forward AllGather
        ("ReduceScatter", "X", "dW_qkv"),
        ("AllReduce", "Z", "dW_qkv"),
        ("AllReduce", "C", "dW_qkv"),
    ]
    rs = [s for s in attn if isinstance(s, ReduceScatterStep) and s.src.name == "dW_qkv"][0]
    assert rs.src.partial == {"X", "Z", "C"}
    assert rs.out.sharding == {"D": "X", "H": "Y"} and rs.out.partial == {"Z", "C"}


def test_backward_expert_weight_grads_never_reduce_over_the_expert_axis():
    steps = train_steps(CFG)
    moe = colls(steps, layer=2, phase="moe", backward=True)
    assert [sig(s) for s in moe] == [
        ("AllToAll", "Z", "dOut"),
        ("AllGather", "X", "W_out"),
        ("AllGather", "Y", "dX"),
        ("ReduceScatter", "X", "dW_out"),
        ("AllReduce", "C", "dW_out"),
        ("AllGather", "X", "W_in"),
        ("ReduceScatter", "Y", "dX"),
        ("ReduceScatter", "X", "dW_in"),
        ("AllReduce", "C", "dW_in"),
        ("AllToAll", "Z", "dX"),
    ]
    assert not any(isinstance(s, AllReduceStep) and s.axis == "Z" for s in moe)
    final = [s for s in moe if isinstance(s, AllReduceStep) and s.src.name == "dW_in"][0]
    assert final.out.sharding == {"E": "Z", "D": "X", "F": "Y"} and not final.out.partial


def test_totals_and_attribution():
    steps = train_steps(CFG)
    assert count_collectives([s for s in steps if not s.backward]) == {
        "AllGather": 16, "ReduceScatter": 4, "AllToAll": 4, "P2P": 1,
    }
    assert count_collectives([s for s in steps if s.backward]) == {
        "AllGather": 16, "ReduceScatter": 16, "AllReduce": 12, "AllToAll": 4, "P2P": 1,
    }
    mm = count_matmuls(steps)
    assert mm["backward"] == 2 * mm["forward"] == 16
    # every collective names one mesh axis with a parallelism role
    for s in colls(steps):
        axis = "stage" if isinstance(s, P2PSendStep) else s.axis
        assert axis in CFG.axis_roles
    assert len([s for s in steps if isinstance(s, SaveActivationStep)]) == 16
