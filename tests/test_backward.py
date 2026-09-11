"""Pin the backward pass's collective sequences and the 2x compute ratio."""

from tpviz import configs
from tpviz.core.backward import count_matmuls, train_steps
from tpviz.core.steps import (
    AllGatherStep,
    AllReduceStep,
    AllToAllStep,
    CollectiveStep,
    GradInitStep,
    P2PSendStep,
    ReduceScatterStep,
    SaveActivationStep,
    StageComputeStep,
)


def bwd_collectives(steps):
    return [s for s in steps if isinstance(s, CollectiveStep) and s.backward]


def test_backward_is_twice_the_forward_compute():
    for cfg in (configs.DP, configs.FSDP, configs.TP, configs.CP, configs.EP):
        counts = count_matmuls(train_steps(cfg))
        assert counts["backward"] == 2 * counts["forward"], cfg.short


def test_forward_saves_activations():
    steps = train_steps(configs.DP)
    saves = [s for s in steps if isinstance(s, SaveActivationStep)]
    # per layer: 4 matmul inputs + Q + K + V = 7
    assert len(saves) == 14
    grad_init = [s for s in steps if isinstance(s, GradInitStep)]
    assert len(grad_init) == 1 and grad_init[0].out.name == "dOut"
    # every save happens before the backward pass starts
    init_idx = steps.index(grad_init[0])
    assert all(steps.index(s) < init_idx for s in saves)


def test_dp_backward_allreduces_each_weight_grad():
    steps = train_steps(configs.DP)
    cs = bwd_collectives(steps)
    assert all(isinstance(s, AllReduceStep) for s in cs)
    assert [s.src.name for s in cs if s.layer == 2] == ["dW_out", "dW_in", "dW_o", "dW_qkv"]
    assert len(cs) == 8 and all(s.axis == "X" for s in cs)


def test_fsdp_backward_regathers_weights_and_reduce_scatters_grads():
    steps = train_steps(configs.FSDP)
    cs = bwd_collectives(steps)
    ags = [s for s in cs if isinstance(s, AllGatherStep)]
    rss = [s for s in cs if isinstance(s, ReduceScatterStep)]
    assert len(ags) == 8 and all(a.jit and a.src.kind == "weight" for a in ags)
    assert len(rss) == 8 and all(r.src.name.startswith("dW_") for r in rss)
    assert all(r.scatter_dim == "D" and r.axis == "X" for r in rss)
    # gradients end up sharded exactly like the weights (ZeRO-3)
    assert all(r.out.sharding == {"D": "X"} for r in rss)


def test_tp_backward_mirrors_forward():
    steps = train_steps(configs.TP)
    layer2 = [s for s in bwd_collectives(steps) if s.layer == 2]
    kinds = [(type(s).__name__, s.src.name) for s in layer2]
    assert kinds == [
        ("AllGatherStep", "dOut"),   # gather the incoming grad (bwd of the fwd ReduceScatter)
        ("ReduceScatterStep", "dX"),  # scatter dX (bwd of the fwd AllGather)
        ("AllGatherStep", "dX"),
        ("ReduceScatterStep", "dX"),
    ]
    # weight grads are local in pure TP
    assert not any(s.src.name.startswith("dW_") for s in bwd_collectives(steps))


def test_cp_backward_scatters_kv_grads_and_allreduces_weight_grads():
    steps = train_steps(configs.CP)
    layer2 = [s for s in bwd_collectives(steps) if s.layer == 2]
    rss = [s for s in layer2 if isinstance(s, ReduceScatterStep)]
    ars = [s for s in layer2 if isinstance(s, AllReduceStep)]
    assert [(r.src.name, r.scatter_dim) for r in rss] == [("dK", "T"), ("dV", "T")]
    assert [a.src.name for a in ars] == ["dW_out", "dW_in", "dW_o", "dW_qkv"]
    assert all(s.axis == "X" for s in layer2)


def test_ep_backward_alltoalls_token_grads():
    steps = train_steps(configs.EP)
    cs = bwd_collectives(steps)
    a2a = [s for s in cs if isinstance(s, AllToAllStep)]
    assert [(s.direction, s.layer) for s in a2a] == [
        ("dispatch", 2), ("combine", 2), ("dispatch", 1), ("combine", 1),
    ]
    # attention weight grads all-reduce over the batch axis Z
    ars = [s for s in cs if isinstance(s, AllReduceStep)]
    assert [a.src.name for a in ars if a.layer == 2] == ["dW_o", "dW_qkv"]
    assert all(a.axis == "Z" for a in ars)


def test_pp_backward_schedule_double_width_and_reverse_sends():
    steps = train_steps(configs.PP)
    bwd_computes = [s for s in steps if isinstance(s, StageComputeStep) and s.backward]
    bwd_sends = [s for s in steps if isinstance(s, P2PSendStep) and s.backward]
    assert len(bwd_computes) == 8 and len(bwd_sends) == 4
    # backward starts at the LAST stage and sends toward stage 0
    assert all(p.dst_stage == p.src_stage - 1 for p in bwd_sends)
    first = min(bwd_computes, key=lambda c: c.tick)
    assert first.stage == 1 and first.microbatch == 3
    # backward cells are 2 ticks apart (2x the forward compute)
    s1_ticks = sorted(c.tick for c in bwd_computes if c.stage == 1)
    assert all(b - a == 2 for a, b in zip(s1_ticks, s1_ticks[1:]))
