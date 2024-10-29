


from ..utils import DummyObject, requires_backends


class LDMTextToImagePipeline(metaclass=DummyObject):
    _backends = ["torch", "transformers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "transformers"])


class StableDiffusionImg2ImgPipeline(metaclass=DummyObject):
    _backends = ["torch", "transformers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "transformers"])


class StableDiffusionInpaintPipeline(metaclass=DummyObject):
    _backends = ["torch", "transformers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "transformers"])


class StableDiffusionPipeline(metaclass=DummyObject):
    _backends = ["torch", "transformers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "transformers"])
