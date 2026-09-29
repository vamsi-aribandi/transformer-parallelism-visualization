"""Logical tensors: dims + sharding + partial-sum axes, independent of rendering."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Mapping

from tpviz import notation
from tpviz.core.mesh import canon_axes

Kind = Literal["weight", "activation", "kv", "grad"]


@dataclass(frozen=True)
class LTensor:
    name: str
    dims: tuple[str, ...]
    sharding: Mapping[str, str] = field(default_factory=dict)  # dim -> mesh axis letters
    partial: frozenset[str] = frozenset()  # unreduced mesh axes (letters), renders {U_X}
    kind: Kind = "activation"

    def __post_init__(self) -> None:
        # plain dict (not MappingProxyType): mobjects holding an LTensor must
        # survive Manim's deepcopy, and mappingproxy cannot be pickled
        object.__setattr__(
            self, "sharding", {d: canon_axes(a) for d, a in self.sharding.items() if a}
        )
        for dim in self.sharding:
            if dim not in self.dims:
                raise ValueError(f"{self.name}: sharded dim {dim!r} not in dims {self.dims}")
        letters = [a for axes in self.sharding.values() for a in axes]
        if len(letters) != len(set(letters)):
            raise ValueError(f"{self.name}: a mesh axis shards more than one dim: {dict(self.sharding)}")

    def axis_of(self, dim: str) -> str | None:
        """The (possibly compound) subscript of `dim`, or None if replicated."""
        return self.sharding.get(dim)

    def axes_of(self, dim: str) -> str:
        """The letters sharding `dim` ("" if replicated)."""
        return self.sharding.get(dim, "")

    def gathered(self, dim: str, axis: str | None = None) -> LTensor:
        """Result of AllGather along `axis` (one letter) — or along every axis
        sharding `dim` when `axis` is None. The subscript loses that letter."""
        new = dict(self.sharding)
        if axis is None:
            new.pop(dim, None)
        else:
            left = new.get(dim, "").replace(axis, "")
            if left:
                new[dim] = left
            else:
                new.pop(dim, None)
        return replace(self, sharding=new)

    def scattered(self, dim: str, axis: str) -> LTensor:
        """Result of ReduceScatter: partial axis resolved, `dim` now (also) sharded on it."""
        new = dict(self.sharding)
        new[dim] = canon_axes(new.get(dim, "") + axis)
        return replace(self, sharding=new, partial=self.partial - {axis})

    def reduced(self, axis: str) -> LTensor:
        """Result of AllReduce: partial axis resolved, sharding unchanged."""
        return replace(self, partial=self.partial - {axis})

    def renamed(self, name: str) -> LTensor:
        return replace(self, name=name)

    def transposed(self) -> LTensor:
        """Swap the last two dims (sharding follows the dims, so it's untouched)."""
        dims = self.dims[:-2] + (self.dims[-1], self.dims[-2])
        return replace(self, dims=dims)

    def grad(self) -> LTensor:
        """The gradient tensor: same dims/sharding, name d<name>, kind 'grad'."""
        return replace(self, name=f"d{self.name}", kind="grad", partial=frozenset())

    def tex(self) -> str:
        return notation.tensor_tex(self)
