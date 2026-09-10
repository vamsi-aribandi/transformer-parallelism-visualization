from manim import (
    DOWN,
    RIGHT,
    UP,
    Arc,
    Dot,
    FadeIn,
    FadeOut,
    RoundedRectangle,
    Text,
    VGroup,
)

from tpviz import configs, style
from tpviz.core.steps import AllGatherStep, AnnotateStep, Step
from tpviz.scenes.base import ForwardPassScene


class CPScene(ForwardPassScene):
    cfg = configs.CP

    def play_step(self, step: Step):
        super().play_step(step)
        if step.layer == 1 and isinstance(step, AllGatherStep) and step.src.name == "V":
            self.ring_attention_callout()

    def on_phase(self, step: Step):
        super().on_phase(step)
        if step.layer == 1 and step.phase == "mlp":
            self.play_step(
                AnnotateStep(
                    text="Tokens are independent in the MLP — no communication here",
                    layer=1,
                    phase="mlp",
                )
            )

    def ring_attention_callout(self):
        """Inset: the optimized variant passes K/V chunks around a ring."""
        box = RoundedRectangle(corner_radius=0.15, width=5.4, height=2.4, stroke_width=1.6)
        box.set_fill("#1c2431", opacity=0.97)
        box.set_stroke(style.ACCENT, opacity=0.8)

        dots = VGroup(
            *[
                Dot(radius=0.09, color=style.DEVICE_HUES[i])
                for i in range(4)
            ]
        )
        import numpy as np

        r = 0.55
        for i, d in enumerate(dots):
            a = i * np.pi / 2
            d.move_to([r * np.cos(a), r * np.sin(a), 0])
        arrows = VGroup(
            *[
                Arc(radius=r + 0.18, start_angle=i * np.pi / 2 + 0.25, angle=np.pi / 2 - 0.5,
                    stroke_width=2.2).add_tip(tip_length=0.12, tip_width=0.12)
                for i in range(4)
            ]
        )
        arrows.set_color(style.KV_STROKE)
        ring = VGroup(dots, arrows)

        text = Text(
            "Optimized variant: ring attention.\nK/V chunks pass around a ring,\noverlapping with compute.",
            font_size=17,
            color=style.TEXT_COLOR,
            line_spacing=0.9,
        )
        content = VGroup(ring, text).arrange(RIGHT, buff=0.5)
        content.move_to(box.get_center())
        callout = VGroup(box, content)
        callout.to_corner(UP + RIGHT, buff=0.4).shift(DOWN * 0.75)
        callout.set_z_index(style.Z_STRIP + 2)

        self.play(FadeIn(callout, scale=0.92), run_time=self.rt(0.6))
        self.wait(self.rt(2.2))
        self.play(FadeOut(callout), run_time=self.rt(0.5))
