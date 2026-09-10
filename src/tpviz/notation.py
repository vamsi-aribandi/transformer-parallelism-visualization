r"""All scaling-book LaTeX notation is built here — the single choke point.

Conventions (from https://jax-ml.github.io/scaling-book/sharding/):
  - A[I_X, J]      dim I sharded over mesh axis X; no subscript = replicated
  - A[I_XY, J]     dim I sharded over axes X and Y
  - C[I, K]{U_X}   partial (unreduced) sum along axis X
  - AllGather_X A[I_X, J] -> A[I, J]
  - ReduceScatter_X,K C[I, K]{U_X} -> C[I, K_X]
  - AllReduce_X C[I, K]{U_X} -> C[I, K]
  - AllToAll_X,J A[I_X, J] -> A[I, J_X]

MathTex pitfalls handled here: literal braces escaped as \{ \}, multi-char
subscripts wrapped in {...}.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tpviz.core.tensors import LTensor

NAME_TEX = {
    "In": r"\mathrm{In}",
    "Out": r"\mathrm{Out}",
    "X": r"X",
    "QKV": r"\mathrm{QKV}",
    "Q": "Q",
    "K": "K",
    "V": "V",
    "A": "A",
    "Tmp": r"\mathrm{Tmp}",
    "W_qkv": r"W_{\mathrm{qkv}}",
    "W_o": r"W_{\mathrm{o}}",
    "W_in": r"W_{\mathrm{in}}",
    "W_out": r"W_{\mathrm{out}}",
}


def name_tex(name: str) -> str:
    if name in NAME_TEX:
        return NAME_TEX[name]
    if len(name) == 1:
        return name
    return rf"\mathrm{{{name}}}"


def dim_tex(dim: str, axes: str | None = None) -> str:
    """"B" -> "B"; ("B", "X") -> "B_{X}"; ("B", "XY") -> "B_{XY}"."""
    if not axes:
        return dim
    return f"{dim}_{{{axes}}}"


def tensor_tex(t: LTensor) -> str:
    """Full sharded-array notation, e.g. \\mathrm{In}[B_{X}, T, D]\\{U_{Y}\\}."""
    dims = ",\\, ".join(dim_tex(d, t.sharding.get(d)) for d in t.dims)
    tex = rf"{name_tex(t.name)}[{dims}]"
    for axis in sorted(t.partial):
        tex += rf"\{{U_{{{axis}}}\}}"
    return tex


def op_tex(op: str, axis: str, dim: str | None = None) -> str:
    """AllGather_X or ReduceScatter_{X,K} style operation label."""
    sub = axis if dim is None else f"{axis},{dim}"
    return rf"\text{{{op}}}_{{{sub}}}"


def collective_tex(op: str, axis: str, src: LTensor, out: LTensor, dim: str | None = None) -> str:
    """Strip line: AllGather_X In[B_X, D] -> In[B, D]."""
    return rf"{op_tex(op, axis, dim)}\; {tensor_tex(src)} \,\to\, {tensor_tex(out)}"


def matmul_tex(a: LTensor, b: LTensor, out: LTensor, contract: str) -> str:
    """A[...] \\cdot_D B[...] -> Out[...] (the book's contraction-dim dot)."""
    return rf"{tensor_tex(a)} \cdot_{{{contract}}} {tensor_tex(b)} \,\to\, {tensor_tex(out)}"


def mesh_tex(axes: dict[str, int]) -> str:
    """Mesh({'X': 4, 'Y': 2}) as in the book."""
    inner = ",\\ ".join(rf"\text{{'{name}'}}{{:}}\,{size}" for name, size in axes.items())
    return rf"\text{{Mesh}}(\{{{inner}\}})"
