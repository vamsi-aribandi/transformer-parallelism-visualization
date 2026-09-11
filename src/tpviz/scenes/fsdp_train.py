from tpviz import configs
from tpviz.scenes.train_base import TrainScene


class FSDPTrainScene(TrainScene):
    cfg = configs.FSDP
