import torch
import numpy as np
import cv2
from pytorch_lightning.utilities.distributed import rank_zero_only
import os, sys
from typing import Any, List
from src.general_utils.os_utils import copy_project_files
import pytorch_lightning as pl
from src.diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from src.diffusers.schedulers.scheduling_ddim import DDIMScheduler
from src.diffusers.training_utils import EMAModel
from torch.nn import functional as F
from src.general_utils.op_utils import count_params
from src.models.conditioner import make_condition
from src.models import model_helper
# from src.losses.consistency_loss import calc_identity_consistency_loss                             # original
from src.losses.consistency_loss import calc_identity_consistency_loss, calc_3dmm_consistency_loss   # Bernardo
from src.recognition.external_mapping import make_external_mapping
from src.recognition.label_mapping import make_label_mapping
from src.recognition.recognition_helper import disabled_train
from src.recognition.recognition_helper import RecognitionModel, make_recognition_model, same_config
from src.recognition.reconstruction_helper import ReconstructionModel, make_3d_face_reconstruction_model
import torchmetrics
from functools import partial

from pytorch3d.io import save_ply, save_obj


class TrainerWith3DMMConsistencyConstraints(pl.LightningModule):
    """main class"""

    def __init__(self, datamodule=None, unet_config=None, optimizer=None, paths=None, sampler=None, ckpt_path=None, *args, **kwargs):

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # ex: self.hparams.net.keywords['input_size']
        self.save_hyperparameters(logger=False)
        super(TrainerWith3DMMConsistencyConstraints, self).__init__()
        self.model = model_helper.make_unet(unet_config)
        self.ema_model = EMAModel(self.model, inv_gamma=1.0, power=3/4, max_value=0.9999)
        count_params(self.model, verbose=True)
        if 'gradient_checkpointing' in unet_config['params'] and unet_config['params']['gradient_checkpointing']:
            self.model.enable_gradient_checkpointing()

        self.valid_loss_metric = torchmetrics.MeanMetric()

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=sampler['num_train_timesteps'],
                                             beta_start=sampler['beta_start'],
                                             beta_end=sampler['beta_end'],
                                             variance_type=sampler['variance_type'],
                                             tensor_format="pt")
        self.noise_scheduler_ddim = DDIMScheduler(num_train_timesteps=sampler['num_train_timesteps'],
                                                  beta_start=sampler['beta_start'],
                                                  beta_end=sampler['beta_end'],
                                                  tensor_format="pt")

        # disabled training
        self.recognition_model: RecognitionModel = make_recognition_model(self.hparams.recognition)
        if same_config(self.hparams.recognition, self.hparams.recognition_eval,
                       skip_keys=['return_spatial', 'center_path']):
            self.recognition_model_eval = self.recognition_model
        else:
            self.recognition_model_eval: RecognitionModel = make_recognition_model(self.hparams.recognition_eval)

        # 3D face reconstruction model
        self.reconstruction_model: ReconstructionModel = make_3d_face_reconstruction_model(self.hparams.reconstruction)
        self.reconstruction_model_eval = self.reconstruction_model
        # if same_config(self.hparams.reconstruction, self.hparams.reconstruction_eval, skip_keys=['return_spatial', 'center_path']):
        #     self.reconstruction_model_eval = self.reconstruction_model
        # else:
        #     # self.reconstruction_model_eval: ReconstructionModel = make_3d_face_reconstruction_model(self.hparams.reconstruction_eval)
        #     self.reconstruction_model_eval: ReconstructionModel = make_3d_face_reconstruction_model(self.hparams.reconstruction_eval)

        self.label_mapping = make_label_mapping(self.hparams.label_mapping, self.hparams.unet_config)
        self.external_mapping = make_external_mapping(self.hparams.external_mapping, self.hparams.unet_config)

        if ckpt_path is not None:
            print('loading checkpoint in initalization from ', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']
            model_statedict = {key[6:] :val for key, val in ckpt.items() if key.startswith('model.')}
            self.model.load_state_dict(model_statedict)

        if self.hparams.unet_config['freeze_unet']:
            print('freeze unet')
            self.model = self.model.eval()
            self.model.train = partial(disabled_train, self=self.model)
            for param in self.model.parameters():
                param.requires_grad = False


    def get_parameters(self):
        if self.hparams.unet_config['freeze_unet']:
            print('freeze unet skip optim params')
            params = []
        else:
            params = list(self.model.parameters())
        if self.external_mapping is not None:
            params = params + list(self.external_mapping.parameters())
        if self.label_mapping is not None:
            params = params + list(self.label_mapping.parameters())
        return params


    def configure_optimizers(self):
        opt = self.hparams.optimizer.optimizer_model(params=self.get_parameters())
        return [opt], []
    
    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        result = super(TrainerWith3DMMConsistencyConstraints, self).load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            print('\n\n\n\nMissing Keys during Loading statedict')
            print(result.missing_keys)
        if result.unexpected_keys:
            print('\n\n\n\nunexpected_keys Keys during Loading statedict')
            print(result.unexpected_keys)
        return result


    @property
    def num_samples(self):
        return self.global_step * self.hparams.datamodule.keywords['total_gpu_batch_size']

    @rank_zero_only
    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        print('copying files', self.hparams.paths.root_dir)
        if self.current_epoch == 0:
            # one time copy of project files
            os.makedirs(self.hparams.paths.output_dir, exist_ok=True)
            copy_project_files(self.hparams.paths.root_dir, self.hparams.paths.output_dir)


    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        pass


    def save_batch(self, current_epoch, batch_idx, batch, output_dir):
        dir_save_batch = os.path.join(output_dir, f'batch_samples/epoch_{str(current_epoch).zfill(3)}/batch_{str(batch_idx).zfill(6)}')
        os.makedirs(dir_save_batch, exist_ok=True)
        # print(f'Saving batch {batch_idx}')

        for batch_key in batch.keys():
            if not batch[batch_key] is None:
                if batch_key == 'image' or batch_key == 'orig' or batch_key == 'id_image' or batch_key == 'extra_image' or batch_key == 'extra_orig' \
                or batch_key == 'noisy_images' or batch_key == 'noise_pred' or batch_key == 'x0_pred':
                    for idx_img, img_torch in enumerate(batch[batch_key]):
                        img_rgb = ((img_torch.permute(1, 2, 0).detach().cpu().numpy() + 1) * 127.5).astype(np.uint8)
                        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                        file_name = f'{batch_key}_{str(idx_img).zfill(4)}.png'
                        file_path = os.path.join(dir_save_batch, file_name)
                        cv2.imwrite(file_path, img_bgr)
                elif 'pointcloud' in batch_key:
                    faces = self.reconstruction_model.backbone.render.faces[0].cpu()
                    for idx_pc, pc_torch in enumerate(batch[batch_key]):
                        pc_file_name = f'{batch_key}_{str(idx_pc).zfill(4)}'
                        save_ply(f'{dir_save_batch}/{pc_file_name}.ply', pc_torch, faces=faces)
                        # save_obj(f'{dir_save_batch}/{pc_file_name}.obj', pc_torch, faces=faces)
                else:
                    # print(f"batch[{batch_key}]: {batch[batch_key]}")
                    pass


    def shared_step(self, batch, batch_idx, stage='train', optimizer_idx=0, *args, **kwargs):
        clean_images = batch['image']
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bsz = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
        ).long()

        loss_dict = {}
        total_loss = 0.0

        if optimizer_idx == 0:
            noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
            encoder_hidden_states = self.get_encoder_hidden_states(batch, batch_size=None)
            noise_pred = self.model(noisy_images, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            mse_loss = F.mse_loss(noise_pred, noise)
            total_loss = total_loss + mse_loss * self.hparams.losses.mse_loss_lambda
            loss_dict[f'{stage}/mse_loss'] = mse_loss
            if stage != 'train':
                loss_dict[f'{stage}/total_loss'] = total_loss
                return total_loss, loss_dict

            if encoder_hidden_states is not None and 'vqloss' in encoder_hidden_states:
                vqloss = encoder_hidden_states.pop('vqloss')
                loss_dict[f'{stage}/vq_loss'] = vqloss
                total_loss = total_loss + vqloss

            # extra identity_consistency_loss_lambda
            if self.hparams.losses.identity_consistency_loss_lambda > 0 or \
                    self.hparams.losses.spatial_consistency_loss_lambda > 0:
                id_loss, spatial_loss = calc_identity_consistency_loss(eps=noise_pred, timesteps=timesteps,
                                                                       noisy_images=noisy_images, batch=batch,
                                                                       pl_module=self)
                total_loss = total_loss + id_loss * self.hparams.losses.identity_consistency_loss_lambda
                if spatial_loss is not None:
                    total_loss = total_loss + spatial_loss * self.hparams.losses.spatial_consistency_loss_lambda
                    loss_dict[f'{stage}/spatial_loss'] = spatial_loss
                loss_dict[f'{stage}/id_loss'] = id_loss

            # 3D consistency constraint
            if self.hparams.losses.threeDMM_consistency_loss_lambda > 0:
                threeDMM_loss, \
                x0_pred, id_image, \
                x0_pred_pointcloud,  x0_pred_3dmm, x0_pred_render_image, \
                id_image_pointcloud, id_image_3dmm, id_image_render_image = calc_3dmm_consistency_loss(eps=noise_pred, timesteps=timesteps,
                                                           noisy_images=noisy_images, batch=batch,
                                                           pl_module=self)
                total_loss = total_loss + threeDMM_loss * self.hparams.losses.threeDMM_consistency_loss_lambda
                loss_dict[f'{stage}/3dmm_loss'] = threeDMM_loss

                if batch_idx == 0:
                    # batch.keys(): dict_keys(['image', 'index', 'orig', 'class_label', 'human_label', 'id_image', 'extra_image', 'extra_index', 'extra_orig'])
                    batch['noisy_images'] = noisy_images
                    batch['noise_pred'] = noise_pred
                    batch['x0_pred'] = x0_pred
                    batch['x0_pred_pointcloud'] = x0_pred_pointcloud
                    batch['x0_pred_3dmm'] = x0_pred_3dmm
                    batch['x0_pred_render_image'] = x0_pred_render_image
                    batch['id_image_pointcloud'] = id_image_pointcloud
                    batch['id_image_3dmm'] = id_image_3dmm
                    batch['id_image_render_image'] = id_image_render_image
                    self.save_batch(self.current_epoch, batch_idx, batch, self.hparams.paths.output_dir)
                    # sys.exit(0)

            loss_dict[f'{stage}/total_loss'] = total_loss

        return total_loss, loss_dict

    def get_encoder_hidden_states(self, batch, batch_size=None):
        encoder_hidden_states = make_condition(pl_module=self,
                                               condition_type=self.hparams.unet_config.params['condition_type'],
                                               condition_source=self.hparams.unet_config.params['condition_source'],
                                               batch=batch
                                               )
        if batch_size is not None and encoder_hidden_states is not None:
            for key, val in encoder_hidden_states.items():
                if val is not None:
                    encoder_hidden_states[key] = val[:batch_size]

        return encoder_hidden_states

    def forward(self, x, c, *args, **kwargs):
        raise ValueError('should not be here. Not Implemented')


    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # import cv2
        # from src.general_utils.img_utils import tensor_to_numpy
        # cv2.imwrite('/mckim/temp/temp3.png',tensor_to_numpy(batch['image'].cpu()[10])) # this is in rgb. so wrong color saved

        loss, loss_dict = self.shared_step(batch, batch_idx, stage='train', optimizer_idx=optimizer_idx)
        if self.hparams.use_ema:
            if self.ema_model.averaged_model.device != self.device:
                self.ema_model.averaged_model.to(self.device)
            self.ema_model.step(self.model)
            self.log("ema_decay", self.ema_model.decay, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        # self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(loss_dict, prog_bar=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, stage='val'):
        _, loss_dict = self.shared_step(batch, batch_idx, stage=stage)
        self.valid_loss_metric.update(loss_dict[f'{stage}/mse_loss'])


    def validation_epoch_end(self, outputs, stage='val', *args, **kwargs):
        self.log('num_samples', self.num_samples)
        self.log('epoch', self.current_epoch)
        self.log('global_step', self.global_step)
        self.log(f'{stage}/mse_loss', self.valid_loss_metric.compute())
        self.valid_loss_metric.reset()


    def on_train_batch_end(self, *args, **kwargs):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        return self.validation_step(batch, batch_idx, stage='test')

    def test_epoch_end(self, outputs: List[Any]):
        return self.validation_epoch_end(outputs, stage='test')

    @property
    def x_T_size(self):
        in_channels = self.hparams.unet_config.params.in_channels
        encoded_size = self.hparams.unet_config.params.image_size
        x_T_size = [in_channels, encoded_size, encoded_size]
        return x_T_size
