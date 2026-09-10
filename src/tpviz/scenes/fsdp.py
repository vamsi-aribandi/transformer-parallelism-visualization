from tpviz import configs
from tpviz.scenes.base import ForwardPassScene


class FSDPScene(ForwardPassScene):
    cfg = configs.FSDP
