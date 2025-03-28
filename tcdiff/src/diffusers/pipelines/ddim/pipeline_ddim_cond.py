















import warnings
from typing import Optional, Tuple, Union

import torch

from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput


class DDIMPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        encoder_hidden_states=None,
        return_x0_intermediates=False,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                The eta parameter which controls the scale of the variance (0 is DDIM and 1 is one type of DDPM).
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """

        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )


            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)




        image = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size),
            generator=generator,
        )
        image = image.to(self.device)


        self.scheduler.set_timesteps(num_inference_steps)

        x0_intermediates = {}
        for t in self.progress_bar(self.scheduler.timesteps):

            model_output = self.unet(image, t, encoder_hidden_states=encoder_hidden_states).sample
            if return_x0_intermediates:
                x0_pred_vis = self.scheduler.step(model_output, t, image, eta).pred_original_sample
                x0_pred_vis = (x0_pred_vis.clone() / 2 + 0.5).clamp(0, 1)
                x0_pred_vis = x0_pred_vis.cpu().permute(0, 2, 3, 1).numpy()
                x0_intermediates[t] = x0_pred_vis



            image = self.scheduler.step(model_output, t, image, eta).prev_sample

            if 'ilvr' in kwargs:
                (down, up, range_t, ref_img) = kwargs['ilvr']
                if t > range_t:
                    d_ref = self.scheduler.add_noise(ref_img, torch.randn(*image.shape, device=image.device), t)
                    image = image - up(down(image)) + up(down(d_ref))


        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)


        return ImagePipelineOutput(images=image, x0_intermediates=x0_intermediates)
