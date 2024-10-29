


from ..utils import DummyObject, requires_backends


class FlaxStableDiffusionPipeline(metaclass=DummyObject):
    _backends = ["flax", "transformers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax", "transformers"])
