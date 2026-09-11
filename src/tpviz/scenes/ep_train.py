from tpviz import configs
from tpviz.scenes.ep import MoETokenMixin
from tpviz.scenes.train_base import TrainScene


class EPTrainScene(MoETokenMixin, TrainScene):
    cfg = configs.EP
