
















import os, sys
import subprocess
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

import datasets
from utils import util
from utils.best_model import BestModel


class ValidatorMultitaskFacerecognition1(object):
    def __init__(self, trainer):
        self.trainer = trainer
        self.device = self.trainer.device
        self.nfc = self.trainer.nfc
        self.cfg = deepcopy(self.trainer.cfg)
        self.device = trainer.device







        self.labels_map = {}

        self.embeddings_lyhm = {}
        self.best_model = BestModel(trainer)
        self.prepare_data()

    def set_labels_map(self, labels_map):
        self.labels_map = labels_map

    def prepare_data(self):

        self.val_dataset, total_images = datasets.build_val_multitask_facerecognition(self.cfg.dataset, self.device, self.cfg)

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False)

        self.val_iter = iter(self.val_dataloader)
        logger.info(f'[VALIDATOR] Validation dataset is ready with {len(self.val_dataset)} actors and {total_images} images.')

    def state_dict(self):
        return {
            'embeddings_lyhm': self.embeddings_lyhm,
            'best_model': self.best_model.state_dict(),
        }

    def load_state_dict(self, dict):
        self.embeddings_lyhm = dict['embeddings_lyhm']
        self.best_model.load_state_dict(dict['best_model'])

    def update_embeddings(self, actors, arcface):
        B = len(actors)
        for i in range(B):
            actor = actors[i]
            if actor not in self.embeddings_lyhm:
                self.embeddings_lyhm[actor] = []
            self.embeddings_lyhm[actor].append(arcface[i].data.cpu().numpy())

    def run(self):
        with torch.no_grad():

            if self.trainer.device != 0:
                return

            self.nfc.eval()
            optdicts = []
            while True:
                try:
                    batch = next(self.val_iter)
                except Exception as e:
                    print(e)
                    self.val_iter = iter(self.val_dataloader)
                    break

                actors = batch['imagename']
                dataset = batch['dataset']
                images = batch['image'].cuda()
                images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                arcface = batch['arcface'].cuda()
                arcface = arcface.view(-1, arcface.shape[-3], arcface.shape[-2], arcface.shape[-1]).to(self.device)
                flame = batch['flame']

                codedict = self.nfc.encode(images, arcface)
                codedict['flame'] = flame


                imagename = batch['imagename']
                codedict['imagename'] = imagename
                imagelabel = datasets.get_imagelabel_from_imagename(imagename, self.labels_map)
                y_true = datasets.get_onehotvector_from_imagelabel2(imagelabel, len(list(self.labels_map.keys()))).to(self.device)
                codedict['y_true'] = y_true

                opdict = self.nfc.decode(codedict, self.trainer.epoch)




                losses, metrics = self.nfc.compute_losses(self.cfg, None, None, opdict)


                all_loss = 0.
                losses_key = losses.keys()
                for key in losses_key:
                    all_loss = all_loss + losses[key]
                losses['all_loss'] = all_loss


                loss_pred_verts = losses['pred_verts_shape_canonical_diff']
                loss_class = losses['class_loss']
                loss_all = losses['all_loss']



                optdicts.append((opdict, images, dataset, actors, loss_pred_verts, loss_class, loss_all, metrics))


            weighted_average = 0.
            avg_loss_pred_verts = 0.
            avg_loss_class = 0.
            avg_loss_all = 0.
            avg_per_dataset = {}
            avg_acc = 0.
            for optdict in optdicts:

                opdict, images, dataset, actors, loss_pred_verts, loss_class, loss_all, metric = optdict
                name = dataset[0]
                avg_loss_pred_verts += loss_pred_verts
                avg_loss_class += loss_class
                avg_loss_all += loss_all
                avg_acc += metric['acc']
                if name not in avg_per_dataset:
                    avg_per_dataset[name] = (loss_pred_verts, 1.)
                else:
                    l, i = avg_per_dataset[name]
                    avg_per_dataset[name] = (l + loss_pred_verts, i + 1.)

            avg_loss_pred_verts = avg_loss_pred_verts.item() / len(optdicts)
            avg_loss_class = avg_loss_class.item() / len(optdicts)
            avg_loss_all = avg_loss_all.item() / len(optdicts)
            avg_acc = avg_acc.item() / len(optdicts)

            '''

            loss_info = f"Step: {self.trainer.global_step},  Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} \n"
            loss_info += f'  validation loss (avg_loss_pred_verts)         : {avg_loss_pred_verts:.5f} \n'
            logger.info(loss_info)
            '''


            loss_info = f"\n  Validation - Step: {self.trainer.global_step},  Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} \n"
            for k, v in losses.items():
                loss_info = loss_info + f'  validation {k}: {v:.4f}\n'
                if self.cfg.train.write_summary:
                    self.trainer.writer.add_scalar('val_loss/' + k, v, global_step=self.trainer.global_step)
            loss_info += f'  validation loss (avg_loss_pred_verts)  : {avg_loss_pred_verts:.5f} \n'
            loss_info += f'  validation acc (avg_acc)  : {avg_acc:.5f} \n'
            logger.info(loss_info)
            self.trainer.writer.add_scalar('val_loss/avg_pred_verts_loss', avg_loss_pred_verts, global_step=self.trainer.global_step)
            self.trainer.writer.add_scalar('val_loss/avg_class_loss', avg_loss_class, global_step=self.trainer.global_step)
            self.trainer.writer.add_scalar('val_loss/avg_all_loss', avg_loss_all, global_step=self.trainer.global_step)
            self.trainer.writer.add_scalar('val_loss/avg_acc', avg_acc, global_step=self.trainer.global_step)
            

            self.trainer.writer.add_scalar('val/avg_loss_pred_verts', avg_loss_pred_verts, global_step=self.trainer.global_step)
            for key in avg_per_dataset.keys():
                l, i = avg_per_dataset[key]
                avg = l.item() / i
                self.trainer.writer.add_scalar(f'val/avg_{key}', avg, global_step=self.trainer.global_step)


            smoothed_weighted, smoothed = self.best_model(weighted_average, avg_loss_pred_verts)
            self.trainer.writer.add_scalar(f'val/smoothed_avg', smoothed, global_step=self.trainer.global_step)




            if self.trainer.global_step % (self.cfg.train.val_steps * 5) == 0:
                lyhm_keys = list(self.embeddings_lyhm.keys())
                embeddings = {**{key: self.embeddings_lyhm[key] for key in lyhm_keys}}

                self.embeddings_lyhm = {}


            if self.trainer.global_step % self.cfg.train.val_save_img != 0:

                return avg_loss_pred_verts, avg_loss_class, avg_loss_all, avg_acc

            pred_canonical_shape_vertices = torch.empty(0, 3, 512, 512).cuda()
            flame_verts_shape = torch.empty(0, 3, 512, 512).cuda()
            input_images = torch.empty(0, 3, 224, 224).cuda()

            for i in np.random.choice(range(0, len(optdicts)), size=4, replace=False):

                opdict, images, _, _, _, _, _, _ = optdicts[i]


                n = np.random.randint(0, len(images)) (to avoid errors when the batch has only one sample)

                rendering = self.nfc.render.render_mesh(opdict['pred_canonical_shape_vertices'][n:n + 1, ...])
                pred_canonical_shape_vertices = torch.cat([pred_canonical_shape_vertices, rendering])
                rendering = self.nfc.render.render_mesh(opdict['flame_verts_shape'][n:n + 1, ...])
                flame_verts_shape = torch.cat([flame_verts_shape, rendering])
                input_images = torch.cat([input_images, images[n:n + 1, ...]])

            visdict = {
                "pred_canonical_shape_vertices": pred_canonical_shape_vertices,
                "flame_verts_shape": flame_verts_shape,
                "input": input_images
            }

            savepath = os.path.join(self.cfg.output_dir, self.cfg.train.val_vis_dir, f'{self.trainer.global_step:08}.jpg')
            util.visualize_grid(visdict, savepath, size=512)


            return avg_loss_pred_verts, avg_loss_class, avg_loss_all, avg_acc


    def now(self):
        logger.info(f'[Validator] NoW testing has begun...')

        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        path = f'{root}{self.cfg.output_dir[1:]}/now_test_training/predicted_meshes'


        experiment = self.cfg.output_dir.split('/')[-1]
        checkpoint = os.path.join(experiment, 'model.tar')
        

        cmd = f'{root}/testing/now/template_multitask_facerecognition1.sh {experiment} {checkpoint} now {path}'


        print('now - experiment:', experiment)
        print('now - root:', root)
        print('now - cmd:', cmd)

        subprocess.call(cmd, shell=True)
        errors = np.load(f'{path}/results/_computed_distances.npy', allow_pickle=True, encoding="latin1").item()['computed_distances']
        median = np.median(np.hstack(errors))
        mean = np.mean(np.hstack(errors))
        std = np.std(np.hstack(errors))

        self.best_model.now(median, mean, std)

        self.trainer.writer.add_scalar(f'val/now_mean', mean, global_step=self.trainer.global_step)
        logger.info(f'[Validator] NoW testing has ended...')
