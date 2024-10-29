














import os

from .import_utils import (
    ENV_VARS_TRUE_AND_AUTO_VALUES,
    ENV_VARS_TRUE_VALUES,
    USE_JAX,
    USE_TF,
    USE_TORCH,
    DummyObject,
    is_flax_available,
    is_inflect_available,
    is_modelcards_available,
    is_onnx_available,
    is_scipy_available,
    is_tf_available,
    is_torch_available,
    is_transformers_available,
    is_unidecode_available,
    requires_backends,
)
from .logging import get_logger
from .outputs import BaseOutput


logger = get_logger(__name__)


hf_cache_home = os.path.expanduser(
    os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
)
default_cache_path = os.path.join(hf_cache_home, "diffusers")


CONFIG_NAME = "config.json"
WEIGHTS_NAME = "diffusion_pytorch_model.bin"
FLAX_WEIGHTS_NAME = "diffusion_flax_model.msgpack"
ONNX_WEIGHTS_NAME = "model.onnx"
HUGGINGFACE_CO_RESOLVE_ENDPOINT = "https://huggingface.co"
DIFFUSERS_CACHE = default_cache_path
DIFFUSERS_DYNAMIC_MODULE_NAME = "diffusers_modules"
HF_MODULES_CACHE = os.getenv("HF_MODULES_CACHE", os.path.join(hf_cache_home, "modules"))
