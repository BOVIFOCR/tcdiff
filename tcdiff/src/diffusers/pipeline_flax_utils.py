















import importlib
import inspect
import os
from typing import Dict, List, Optional, Union

import numpy as np

import flax
import PIL
from flax.core.frozen_dict import FrozenDict
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm.auto import tqdm

from .configuration_utils import ConfigMixin
from .modeling_flax_utils import FLAX_WEIGHTS_NAME, FlaxModelMixin
from .schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME, SchedulerMixin
from .utils import CONFIG_NAME, DIFFUSERS_CACHE, BaseOutput, is_transformers_available, logging


if is_transformers_available():
    from transformers import FlaxPreTrainedModel

INDEX_FILE = "diffusion_flax_model.bin"


logger = logging.get_logger(__name__)


LOADABLE_CLASSES = {
    "diffusers": {
        "FlaxModelMixin": ["save_pretrained", "from_pretrained"],
        "SchedulerMixin": ["save_config", "from_config"],
        "FlaxDiffusionPipeline": ["save_pretrained", "from_pretrained"],
    },
    "transformers": {
        "PreTrainedTokenizer": ["save_pretrained", "from_pretrained"],
        "PreTrainedTokenizerFast": ["save_pretrained", "from_pretrained"],
        "FlaxPreTrainedModel": ["save_pretrained", "from_pretrained"],
        "FeatureExtractionMixin": ["save_pretrained", "from_pretrained"],
    },
}

ALL_IMPORTABLE_CLASSES = {}
for library in LOADABLE_CLASSES:
    ALL_IMPORTABLE_CLASSES.update(LOADABLE_CLASSES[library])


class DummyChecker:
    def __init__(self):
        self.dummy = True


def import_flax_or_no_model(module, class_name):
    try:

        class_obj = getattr(module, "Flax" + class_name)
    except AttributeError:

        class_obj = getattr(module, class_name)
    except AttributeError:
        raise ValueError(f"Neither Flax{class_name} nor {class_name} exist in {module}")

    return class_obj


@flax.struct.dataclass
class FlaxImagePipelineOutput(BaseOutput):
    """
    Output class for image pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


class FlaxDiffusionPipeline(ConfigMixin):
    r"""
    Base class for all models.

    [`FlaxDiffusionPipeline`] takes care of storing all components (models, schedulers, processors) for diffusion
    pipelines and handles methods for loading, downloading and saving models as well as a few methods common to all
    pipelines to:

        - enabling/disabling the progress bar for the denoising iteration

    Class attributes:

        - **config_name** ([`str`]) -- name of the config file that will store the class and module names of all
          components of the diffusion pipeline.
    """
    config_name = "model_index.json"

    def register_modules(self, **kwargs):

        from diffusers import pipelines

        for name, module in kwargs.items():

            library = module.__module__.split(".")[0]


            pipeline_dir = module.__module__.split(".")[-2]
            path = module.__module__.split(".")
            is_pipeline_module = pipeline_dir in path and hasattr(pipelines, pipeline_dir)




            if library not in LOADABLE_CLASSES or is_pipeline_module:
                library = pipeline_dir


            class_name = module.__class__.__name__

            register_dict = {name: (library, class_name)}


            self.register_to_config(**register_dict)


            setattr(self, name, module)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], params: Union[Dict, FrozenDict]):

        """
        Save all variables of the pipeline that can be saved and loaded as well as the pipelines configuration file to
        a directory. A pipeline variable can be saved and loaded if its class implements both a save and loading
        method. The pipeline can easily be re-loaded using the `[`~FlaxDiffusionPipeline.from_pretrained`]` class
        method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
        """
        self.save_config(save_directory)

        model_index_dict = dict(self.config)
        model_index_dict.pop("_class_name")
        model_index_dict.pop("_diffusers_version")
        model_index_dict.pop("_module", None)

        for pipeline_component_name in model_index_dict.keys():
            sub_model = getattr(self, pipeline_component_name)
            model_cls = sub_model.__class__

            save_method_name = None

            for library_name, library_classes in LOADABLE_CLASSES.items():
                library = importlib.import_module(library_name)
                for base_class, save_load_methods in library_classes.items():
                    class_candidate = getattr(library, base_class)
                    if issubclass(model_cls, class_candidate):

                        save_method_name = save_load_methods[0]
                        break
                if save_method_name is not None:
                    break


            if isinstance(sub_model, DummyChecker):
                continue

            save_method = getattr(sub_model, save_method_name)
            expects_params = "params" in set(inspect.signature(save_method).parameters.keys())

            if expects_params:
                save_method(
                    os.path.join(save_directory, pipeline_component_name), params=params[pipeline_component_name]
                )
            else:
                save_method(os.path.join(save_directory, pipeline_component_name))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        r"""
        Instantiate a PyTorch diffusion pipeline from pre-trained pipeline weights.

        The pipeline is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated).

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *repo id* of a pretrained pipeline hosted inside a model repo on
                      https://huggingface.co/ Valid repo ids have to be located under a user or organization name, like
                      `CompVis/ldm-text2im-large-256`.
                    - A path to a *directory* containing pipeline weights saved using
                      [`~FlaxDiffusionPipeline.save_pretrained`], e.g., `./my_pipeline_directory/`.
            dtype (`str` or `jnp.dtype`, *optional*):
                Override the default `jnp.dtype` and load the model under this dtype. If `"auto"` is passed the dtype
                will be automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information. specify the folder name here.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load - and saveable variables - *i.e.* the pipeline components - of the
                specific pipeline class. The overwritten components are then directly passed to the pipelines
                `__init__` method. See example below for more information.

        <Tip>

        Passing `use_auth_token=True`` is required when you want to use a private model, *e.g.*
        `"CompVis/stable-diffusion-v1-4"`

        </Tip>

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use
        this method in a firewalled environment.

        </Tip>

        Examples:

        ```py
        >>> from diffusers import FlaxDiffusionPipeline

        >>> # Download pipeline from huggingface.co and cache.
        >>> pipeline = FlaxDiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")

        >>> # Download pipeline that requires an authorization token
        >>> # For more information on access tokens, please refer to this section
        >>> # of the documentation](https://huggingface.co/docs/hub/security-tokens)
        >>> pipeline = FlaxDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True)

        >>> # Download pipeline, but overwrite scheduler
        >>> from diffusers import LMSDiscreteScheduler

        >>> scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
        >>> pipeline = FlaxDiffusionPipeline.from_pretrained(
        ...     "CompVis/stable-diffusion-v1-4", scheduler=scheduler, use_auth_token=True
        ... )
        ```
        """
        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        from_pt = kwargs.pop("from_pt", False)
        dtype = kwargs.pop("dtype", None)



        if not os.path.isdir(pretrained_model_name_or_path):
            config_dict = cls.get_config_dict(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
            )

            folder_names = [k for k in config_dict.keys() if not k.startswith("_")]
            allow_patterns = [os.path.join(k, "*") for k in folder_names]
            allow_patterns += [FLAX_WEIGHTS_NAME, SCHEDULER_CONFIG_NAME, CONFIG_NAME, cls.config_name]


            cached_folder = snapshot_download(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                allow_patterns=allow_patterns,
            )
        else:
            cached_folder = pretrained_model_name_or_path

        config_dict = cls.get_config_dict(cached_folder)



        if cls != FlaxDiffusionPipeline:
            pipeline_class = cls
        else:
            diffusers_module = importlib.import_module(cls.__module__.split(".")[0])
            pipeline_class = getattr(diffusers_module, config_dict["_class_name"])




        expected_modules = set(inspect.signature(pipeline_class.__init__).parameters.keys())
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}

        init_dict, _ = pipeline_class.extract_init_dict(config_dict, **kwargs)

        init_kwargs = {}


        params = {}


        from diffusers import pipelines


        for name, (library_name, class_name) in init_dict.items():

            if class_name == "DummyChecker":
                library_name = "stable_diffusion"
                class_name = "FlaxStableDiffusionSafetyChecker"

            is_pipeline_module = hasattr(pipelines, library_name)
            loaded_sub_model = None


            if name in passed_class_obj:

                if not is_pipeline_module:
                    library = importlib.import_module(library_name)
                    class_obj = getattr(library, class_name)
                    importable_classes = LOADABLE_CLASSES[library_name]
                    class_candidates = {c: getattr(library, c) for c in importable_classes.keys()}

                    expected_class_obj = None
                    for class_name, class_candidate in class_candidates.items():
                        if issubclass(class_obj, class_candidate):
                            expected_class_obj = class_candidate

                    if not issubclass(passed_class_obj[name].__class__, expected_class_obj):
                        raise ValueError(
                            f"{passed_class_obj[name]} is of type: {type(passed_class_obj[name])}, but should be"
                            f" {expected_class_obj}"
                        )
                else:
                    logger.warn(
                        f"You have passed a non-standard module {passed_class_obj[name]}. We cannot verify whether it"
                        " has the correct type"
                    )


                loaded_sub_model = passed_class_obj[name]
            elif is_pipeline_module:
                pipeline_module = getattr(pipelines, library_name)
                if from_pt:
                    class_obj = import_flax_or_no_model(pipeline_module, class_name)
                else:
                    class_obj = getattr(pipeline_module, class_name)

                importable_classes = ALL_IMPORTABLE_CLASSES
                class_candidates = {c: class_obj for c in importable_classes.keys()}
            else:

                library = importlib.import_module(library_name)
                if from_pt:
                    class_obj = import_flax_or_no_model(library, class_name)
                else:
                    class_obj = getattr(library, class_name)

                importable_classes = LOADABLE_CLASSES[library_name]
                class_candidates = {c: getattr(library, c) for c in importable_classes.keys()}

            if loaded_sub_model is None:
                load_method_name = None
                for class_name, class_candidate in class_candidates.items():
                    if issubclass(class_obj, class_candidate):
                        load_method_name = importable_classes[class_name][1]

                load_method = getattr(class_obj, load_method_name)


                if os.path.isdir(os.path.join(cached_folder, name)):
                    loadable_folder = os.path.join(cached_folder, name)
                else:
                    loaded_sub_model = cached_folder

                if issubclass(class_obj, FlaxModelMixin):
                    loaded_sub_model, loaded_params = load_method(loadable_folder, from_pt=from_pt, dtype=dtype)
                    params[name] = loaded_params
                elif is_transformers_available() and issubclass(class_obj, FlaxPreTrainedModel):

                    if name == "safety_checker":
                        loaded_sub_model = DummyChecker()
                        loaded_params = {}
                    elif from_pt:

                        loaded_sub_model = load_method(loadable_folder, from_pt=from_pt)
                        loaded_params = loaded_sub_model.params
                        del loaded_sub_model._params
                    else:
                        loaded_sub_model, loaded_params = load_method(loadable_folder, _do_init=False)
                    params[name] = loaded_params
                elif issubclass(class_obj, SchedulerMixin):
                    loaded_sub_model, scheduler_state = load_method(loadable_folder)
                    params[name] = scheduler_state
                else:
                    loaded_sub_model = load_method(loadable_folder)

            init_kwargs[name] = loaded_sub_model  # UNet(...), # DiffusionSchedule(...)

        model = pipeline_class(**init_kwargs, dtype=dtype)
        return model, params

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images


    def progress_bar(self, iterable):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        return tqdm(iterable, **self._progress_bar_config)

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs
