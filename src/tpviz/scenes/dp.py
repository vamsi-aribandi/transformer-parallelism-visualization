from tpviz import configs
from tpviz.core.steps import AnnotateStep
from tpviz.scenes.base import ForwardPassScene


class DPScene(ForwardPassScene):
    cfg = configs.DP

    def summary(self):
        self.speed = 1.0
        self.play_step(
            AnnotateStep(
                text="Forward pass: zero communication. Full weights + a batch shard on every device.",
                layer=2,
                phase="mlp",
            )
        )
        super().summary()
