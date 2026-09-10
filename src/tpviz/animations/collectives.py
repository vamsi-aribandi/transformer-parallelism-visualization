"""Collective-communication animations.

Discipline: collectives animate COPIES of resident shards (never the residents
themselves), give them Z_FLYING while airborne, and clean them up. The caller
supplies both the resident per-device mobjects and the prebuilt, prepositioned
result mobjects; these functions choreograph the exchange and swap them in.
"""

from __future__ import annotations

from manim import (
    AnimationGroup,
    FadeIn,
    FadeOut,
    Mobject,
    ReplacementTransform,
    Scene,
    VGroup,
)

from tpviz import style


def _flight(src: Mobject, dest_center, arc: float = 0.55) -> tuple[Mobject, AnimationGroup]:
    c = src.copy()
    c.set_z_index(style.Z_FLYING)
    anim = c.animate(path_arc=arc).move_to(dest_center).scale(0.85)
    return c, anim


def animate_all_gather(
    scene: Scene,
    shards: list[Mobject],
    results: list[Mobject],
    *,
    run_time: float = 1.5,
) -> None:
    """Every device's shard flies to every other device, tiling the full tensor."""
    copies, flights = [], []
    for j, result in enumerate(results):
        dest = result.get_center()
        for i, shard in enumerate(shards):
            if i == j:
                continue
            c, anim = _flight(shard, dest)
            copies.append(c)
            flights.append(anim)
    scene.add(*copies)
    scene.play(*flights, run_time=run_time * 0.65)
    scene.play(
        FadeOut(VGroup(*copies), run_time=run_time * 0.35),
        *[ReplacementTransform(shards[j], results[j], run_time=run_time * 0.35)
          for j in range(len(shards))],
    )


def animate_exchange_resolve(
    scene: Scene,
    partials: list[Mobject],
    results: list[Mobject],
    *,
    run_time: float = 1.5,
) -> None:
    """ReduceScatter / AllReduce: partial sums crossfly, then resolve.

    The visual: every device sends a copy toward every other device (the
    reduction exchange), then each dashed partial transforms into its solid
    result (a slice for ReduceScatter, the full tensor for AllReduce).
    """
    copies, flights = [], []
    for j, result in enumerate(results):
        dest = result.get_center()
        for i, partial in enumerate(partials):
            if i == j:
                continue
            c, anim = _flight(partial, dest, arc=0.35)
            c.set_opacity(0.35)
            copies.append(c)
            flights.append(anim)
    scene.add(*copies)
    scene.play(*flights, run_time=run_time * 0.6)
    scene.play(
        FadeOut(VGroup(*copies), run_time=run_time * 0.4),
        *[ReplacementTransform(partials[j], results[j], run_time=run_time * 0.4)
          for j in range(len(partials))],
    )


def animate_all_to_all(
    scene: Scene,
    token_groups: list[list[Mobject]],
    destinations: list[list],
    *,
    run_time: float = 1.8,
) -> None:
    """token_groups[i][k] flies to destinations[i][k] (a point). Pure crossfly;
    the caller owns what the tokens mean and what happens after arrival."""
    flights = []
    for tokens, dests in zip(token_groups, destinations):
        for tok, dest in zip(tokens, dests):
            tok.set_z_index(style.Z_FLYING)
            flights.append(tok.animate(path_arc=0.45).move_to(dest))
    scene.play(*flights, run_time=run_time)


def animate_p2p(
    scene: Scene,
    tensor: Mobject,
    dest_center,
    *,
    run_time: float = 1.0,
) -> Mobject:
    """Point-to-point send: one copy slides from src to dst. Returns the copy."""
    c = tensor.copy()
    c.set_z_index(style.Z_FLYING)
    scene.play(c.animate(path_arc=-0.3).move_to(dest_center), run_time=run_time)
    return c
