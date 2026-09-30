"""ZeRO-1 and the two frontier recipes (dense 4D, open-MoE 3D)."""

from tpviz import configs
from tpviz.core.backward import count_matmuls, train_steps
from tpviz.core.model import count_collectives, forward_steps
from tpviz.core.steps import (
    AllGatherStep, AllReduceStep, CollectiveStep, OptimizerStep, P2PSendStep, ReduceScatterStep,
)


def sig(s):
    return (type(s).__name__.removesuffix("Step"), s.axis, s.src.name)


def test_zero1_forward_is_plain_data_parallelism():
    steps = forward_steps(configs.ZERO1)
    assert not [s for s in steps if isinstance(s, CollectiveStep)]


def test_zero1_backward_splits_the_allreduce_around_the_optimizer_step():
    steps = train_steps(configs.ZERO1)
    bwd = [s for s in steps if s.backward]
    rss = [s for s in bwd if isinstance(s, ReduceScatterStep)]
    assert len(rss) == 8 and all(r.axis == "X" and r.src.name.startswith("dW") for r in rss)
    assert not any(isinstance(s, AllReduceStep) for s in bwd)
    upd = [s for s in bwd if isinstance(s, OptimizerStep)]
    ags = [s for s in bwd if isinstance(s, AllGatherStep)]
    assert len(upd) == 8 and len(ags) == 8
    # the updated shards gather back into whole, replicated weights
    assert all(a.src.kind == "weight" and not a.jit and a.out.sharding == {} for a in ags)
    assert all(u.out.sharding == a.src.sharding for u, a in zip(upd, ags))
    # every optimizer step comes after the whole backward pass
    first_upd = steps.index(upd[0])
    assert all(steps.index(r) < first_upd for r in rss)
    assert count_matmuls(steps)["backward"] == 2 * count_matmuls(steps)["forward"]


def test_dense4d_is_llama3_style():
    cfg = configs.DENSE4D
    assert cfg.moe is None and cfg.mesh.n_devices == 16
    steps = train_steps(cfg)
    fwd = [sig(s) for s in steps if isinstance(s, CollectiveStep) and not s.backward and s.layer == 1]
    assert fwd == [
        ("AllGather", "X", "W_qkv"), ("AllGather", "Y", "In"),
        ("AllGather", "C", "K"), ("AllGather", "C", "V"),
        ("AllGather", "X", "W_o"), ("ReduceScatter", "Y", "X"),
        ("AllGather", "X", "W_in"), ("AllGather", "Y", "X"),
        ("AllGather", "X", "W_out"), ("ReduceScatter", "Y", "X"),
    ]
    assert len([s for s in steps if isinstance(s, P2PSendStep)]) == 2
    # weight grads: FSDP scatter over X, then a sum over the context axis; no expert axis
    bwd = [s for s in steps if s.backward and isinstance(s, CollectiveStep) and s.src.name == "dW_qkv"]
    assert [(type(s).__name__, s.axis) for s in bwd if s.layer == 2] == [
        ("ReduceScatterStep", "X"), ("AllReduceStep", "C"),
    ]
    assert count_collectives([s for s in steps if s.backward]) == {
        "AllGather": 16, "ReduceScatter": 16, "AllReduce": 8, "P2P": 1,
    }


def test_moe4d_is_deepseek_style_no_tensor_parallelism():
    cfg = configs.MOE4D
    assert "Y" not in cfg.mesh.axes and cfg.mesh.n_devices == 16
    steps = train_steps(cfg)
    fwd = [s for s in steps if isinstance(s, (CollectiveStep, P2PSendStep)) and not s.backward]
    # forward: no weight or activation gathers except CP's K/V, the AllToAll pairs, the stage hop
    assert count_collectives(fwd) == {"AllGather": 4, "AllToAll": 4, "P2P": 1}
    assert all(s.src.kind == "kv" and s.axis == "C" for s in fwd if isinstance(s, AllGatherStep))
    dispatch = [s for s in fwd if isinstance(s, CollectiveStep) and s.op == "AllToAll"][0]
    assert dispatch.out.sharding == {"E": "Z", "S": "XC"}  # tokens keep their data and context split
    bwd = [s for s in steps if s.backward]
    # attention weight grads: ZeRO-1 scatter over X, then sums over the expert and context axes;
    # expert weight grads: scatter over X, sum over C only (each expert owns its weights)
    attn = [(type(s).__name__, s.axis) for s in bwd if isinstance(s, CollectiveStep)
            and s.layer == 2 and s.src.name == "dW_qkv"]
    assert attn == [("ReduceScatterStep", "X"), ("AllReduceStep", "Z"), ("AllReduceStep", "C")]
    exp = [(type(s).__name__, s.axis) for s in bwd if isinstance(s, CollectiveStep)
           and s.layer == 2 and s.src.name == "dW_in"]
    assert exp == [("ReduceScatterStep", "X"), ("AllReduceStep", "C")]
    upd = [s for s in bwd if isinstance(s, OptimizerStep)]
    assert [u.weight.name for u in upd if u.layer == 1] == ["W_qkv", "W_o", "W_in", "W_out"]
    assert count_collectives(bwd) == {
        "AllToAll": 4, "AllGather": 12, "ReduceScatter": 12, "AllReduce": 12, "P2P": 1,
    }
