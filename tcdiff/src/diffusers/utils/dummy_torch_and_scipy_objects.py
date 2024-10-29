


from ..utils import DummyObject, requires_backends


class LMSDiscreteScheduler(metaclass=DummyObject):
    _backends = ["torch", "scipy"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "scipy"])
