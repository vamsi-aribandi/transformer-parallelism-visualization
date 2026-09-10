"""Logical tensors: dims + sharding + partial-sum axes, independent of rendering."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Mapping

from tpviz import notation

Kind = Literal["weight", "activation", "kv"]


@dataclass(frozen=True)
class LTensor:
    name: str
    dims: tuple[str, ...]
    sharding: Mapping[str, str] = field(default_factory=dict)  # dim -> mesh axis
    partial: frozenset[str] = frozenset()  # unreduced mesh axes, renders {U_X}
    kind: Kind = "activation"

    def __post_init__(self) -> None:
        # plain dict (not MappingProxyType): mobjects holding an LTensor must
        # survive Manim's deepcopy, and mappingproxy cannot be pickled
        object.__setattr__(self, "sharding", dict(self.sharding))
        for dim in self.sharding:
            if dim not in self.dims:
                raise ValueError(f"{self.name}: sharded dim {dim!r} not in dims {self.dims}")
        axes = list(self.sharding.values())
        if len(axes) != len(set(axes)):
            raise ValueError(f"{self.name}: a mesh axis shards more than one dim: {dict(self.sharding)}")

    def axis_of(self, dim: str) -> str | None:
        return self.sharding.get(dim)

    def gathered(self, dim: str) -> LTensor:
        """Result of AllGather along the axis sharding `dim` (subscript removed)."""
        new = {d: a for d, a in self.sharding.items() if d != dim}
        return replace(self, sharding=new)

    def scattered(self, dim: str, axis: str) -> LTensor:
        """Result of ReduceScatter: partial axis resolved, `dim` now sharded on it."""
        new = dict(self.sharding)
        new[dim] = axis
        return replace(self, sharding=new, partial=self.partial - {axis})

    def reduced(self, axis: str) -> LTensor:
        """Result of AllReduce: partial axis resolved, sharding unchanged."""
        return replace(self, partial=self.partial - {axis})

    def renamed(self, name: str) -> LTensor:
        return replace(self, name=name)

    def tex(self) -> str:
        return notation.tensor_tex(self)
