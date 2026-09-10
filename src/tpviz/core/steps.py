"""Forward-pass step IR: semantic, renderer-independent.

Scenes play these back; tests assert the collective sequences match the book.
"""

from __future__ import annotations

from dataclasses import dataclass

from tpviz import notation
from tpviz.core.tensors import LTensor


@dataclass(kw_only=True)
class Step:
    layer: int = 0
    phase: str = ""  # "attn" | "mlp" | "moe" | "pipeline"
    caption: str | None = None
    microbatch: int | None = None


@dataclass(kw_only=True)
class MatMulStep(Step):
    a: LTensor
    b: LTensor
    out: LTensor
    contract: str

    def tex(self) -> str:
        return notation.matmul_tex(self.a, self.b, self.out, self.contract)


@dataclass(kw_only=True)
class CollectiveStep(Step):
    src: LTensor
    out: LTensor
    axis: str

    op: str = ""  # set by subclasses

    def tex(self) -> str:
        return notation.collective_tex(self.op, self.axis, self.src, self.out)


@dataclass(kw_only=True)
class AllGatherStep(CollectiveStep):
    dim: str
    jit: bool = False  # FSDP-style gather-use-discard (weights)
    op: str = "AllGather"


@dataclass(kw_only=True)
class ReduceScatterStep(CollectiveStep):
    scatter_dim: str
    op: str = "ReduceScatter"

    def tex(self) -> str:
        return notation.collective_tex(self.op, self.axis, self.src, self.out, dim=self.scatter_dim)


@dataclass(kw_only=True)
class AllReduceStep(CollectiveStep):
    op: str = "AllReduce"


@dataclass(kw_only=True)
class AllToAllStep(CollectiveStep):
    direction: str = "dispatch"  # "dispatch" | "combine"
    op: str = "AllToAll"


@dataclass(kw_only=True)
class SplitQKVStep(Step):
    src: LTensor
    q: LTensor
    k: LTensor
    v: LTensor


@dataclass(kw_only=True)
class AttentionCoreStep(Step):
    q: LTensor
    k: LTensor
    v: LTensor
    out: LTensor

    def tex(self) -> str:
        return (
            rf"\text{{Attn}}({self.q.tex()},\, {self.k.tex()},\, {self.v.tex()})"
            rf" \,\to\, {self.out.tex()}"
        )


@dataclass(kw_only=True)
class GeluStep(Step):
    src: LTensor
    out: LTensor

    def tex(self) -> str:
        return rf"\text{{gelu}}({self.src.tex()}) \,\to\, {self.out.tex()}"


@dataclass(kw_only=True)
class RouteStep(Step):
    tokens: LTensor
    n_experts: int

    def tex(self) -> str:
        return rf"\text{{Router}}:\ {self.tokens.tex()} \,\to\, \text{{expert assignments}}"


@dataclass(kw_only=True)
class StageComputeStep(Step):
    stage: int
    tick: int


@dataclass(kw_only=True)
class P2PSendStep(Step):
    tensor: LTensor
    src_stage: int
    dst_stage: int
    tick: int

    def tex(self) -> str:
        return (
            rf"\text{{Send}}:\ {self.tensor.tex()}\ \ "
            rf"\text{{stage {self.src_stage}}} \to \text{{stage {self.dst_stage}}}"
        )


@dataclass(kw_only=True)
class AnnotateStep(Step):
    text: str = ""
