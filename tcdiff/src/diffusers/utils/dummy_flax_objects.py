


from ..utils import DummyObject, requires_backends


class FlaxModelMixin(metaclass=DummyObject):
    _backends = ["flax"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax"])


class FlaxUNet2DConditionModel(metaclass=DummyObject):
    _backends = ["flax"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax"])


class FlaxAutoencoderKL(metaclass=DummyObject):
    _backends = ["flax"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax"])


class FlaxDiffusionPipeline(metaclass=DummyObject):
    _backends = ["flax"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax"])


class FlaxDDIMScheduler(metaclass=DummyObject):
    _backends = ["flax"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax"])


class FlaxDDPMScheduler(metaclass=DummyObject):
    _backends = ["flax"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax"])


class FlaxKarrasVeScheduler(metaclass=DummyObject):
    _backends = ["flax"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax"])


class FlaxLMSDiscreteScheduler(metaclass=DummyObject):
    _backends = ["flax"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax"])


class FlaxPNDMScheduler(metaclass=DummyObject):
    _backends = ["flax"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax"])


class FlaxScoreSdeVeScheduler(metaclass=DummyObject):
    _backends = ["flax"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax"])
