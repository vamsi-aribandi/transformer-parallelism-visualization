"""Local-compute animations: matmuls, gelu, the attention capsule."""

from __future__ import annotations

from manim import (
    FadeIn,
    FadeOut,
    Indicate,
    Mobject,
    ReplacementTransform,
    RoundedRectangle,
    Scene,
    Text,
    VGroup,
)

from tpviz import style


def animate_matmul(
    scene: Scene,
    weights: list[Mobject],
    acts: list[Mobject],
    outs: list[Mobject],
    *,
    run_time: float = 1.4,
) -> None:
    """Pulse the operands, then the activation transforms into the product."""
    scene.play(
        *[Indicate(w, scale_factor=1.06, color=style.WEIGHT_STROKE) for w in weights],
        *[Indicate(a, scale_factor=1.06, color=style.ACT_STROKE) for a in acts],
        run_time=run_time * 0.45,
    )
    scene.play(
        *[ReplacementTransform(a, o) for a, o in zip(acts, outs)],
        run_time=run_time * 0.55,
    )


def animate_gelu(scene: Scene, acts: list[Mobject], *, run_time: float = 0.7) -> None:
    scene.play(
        *[Indicate(a, scale_factor=1.08, color=style.GOOD_COLOR) for a in acts],
        run_time=run_time,
    )


def make_attn_capsule(width: float = 1.1, height: float = 0.55) -> VGroup:
    box = RoundedRectangle(corner_radius=0.14, width=width, height=height, stroke_width=2.0)
    box.set_fill("#243041", opacity=1.0)
    box.set_stroke(style.ACCENT, opacity=0.9)
    tag = Text("Attn", font_size=20, color=style.ACCENT, weight="BOLD")
    tag.move_to(box.get_center())
    capsule = VGroup(box, tag)
    capsule.set_z_index(style.Z_TENSOR + 1)
    return capsule


def animate_attention(
    scene: Scene,
    qkv_groups: list[list[Mobject]],  # per device: [Q, K, V] mobjects
    capsules: list[VGroup],  # prepositioned per device
    outs: list[Mobject],  # prepositioned A decks per device
    *,
    run_time: float = 2.0,
) -> None:
    """Q, K, V shrink into an Attn capsule; the capsule yields the A tensor."""
    scene.play(
        *[FadeIn(c, scale=0.8) for c in capsules],
        *[
            m.animate.scale(0.1).move_to(c.get_center()).set_opacity(0.0)
            for group, c in zip(qkv_groups, capsules)
            for m in group
        ],
        run_time=run_time * 0.55,
    )
    for group in qkv_groups:
        for m in group:
            scene.remove(m)
    scene.play(
        *[ReplacementTransform(c, o) for c, o in zip(capsules, outs)],
        run_time=run_time * 0.45,
    )
