


from ..utils import DummyObject, requires_backends


class StableDiffusionOnnxPipeline(metaclass=DummyObject):
    _backends = ["torch", "transformers", "onnx"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "transformers", "onnx"])
