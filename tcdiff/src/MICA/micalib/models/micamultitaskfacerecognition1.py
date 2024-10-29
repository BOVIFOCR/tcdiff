
















import os
import sys

sys.path.append("./nfclib")

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.arcface import Arcface
from models.generator import Generator
from micalib.base_model import BaseModel

from loguru import logger


class ArcFace_MLP(torch.nn.Module):
    def __init__(self, num_classes=None, margin=0.5, scale=64.0, cfg=None, device=None):
        super(ArcFace_MLP, self).__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale


        input_size = self.infer_input_size(self.cfg.model.face_embed)

        self.fc1 = nn.Linear(input_size, 512)
        

        '''
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, self.num_classes)
        self.fc4 = nn.Linear(self.num_classes, self.num_classes)
        self.norm_fc1 = nn.BatchNorm1d(512)
        self.norm_fc2 = nn.BatchNorm1d(1024)
        self.norm_fc3 = nn.BatchNorm1d(self.num_classes)
        '''

        self.weight = nn.Parameter(torch.Tensor(self.num_classes, 512))



        
    def forward(self, x):
        if self.cfg.model.face_embed == 'original-arcface':
            face_embedd = x
            cosine = None
            prob_pred = None
            y_pred = None
        else:
            x = self.fc1(x)
            '''
            x = F.relu(self.norm_fc1(self.fc1(x)))
            x = F.relu(self.norm_fc2(self.fc2(x)))
            x = F.relu(self.norm_fc3(self.fc3(x)))
            x = self.fc4(x)
            '''
            face_embedd = x
            cosine = F.linear(F.normalize(x), F.normalize(self.weight))
            prob_pred = torch.nn.functional.softmax(cosine, dim=1)
            y_pred = torch.argmax(cosine, dim=1)

        return face_embedd, cosine, prob_pred, y_pred

    def infer_input_size(self, face_embed_type='3dmm'):
        if face_embed_type == 'arcface' or face_embed_type == 'original-arcface':
            input_size = 512            # ArcFace embedding (512)
        elif face_embed_type == '3dmm':
            input_size = 300            # 3DMM embedding (300)
        elif face_embed_type == 'arcface-3dmm':
            input_size = 812            # ArcFace (512) + 3DMM embedding (300)
        elif face_embed_type == 'pc_vertices':
            input_size = (5023,3)       # FLAME Point Cloud (5023,3)
        return input_size


class FaceClassifier1_MLP(torch.nn.Module):
    def __init__(self, num_classes=None, model_cfg=None, device=None, tag='FaceClassifier1_MLP'):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(300, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        logits = self.layers(x)
        prob_pred = torch.nn.functional.softmax(logits, dim=1)
        y_pred = torch.argmax(prob_pred, dim=1)
        return logits, prob_pred, y_pred



class MICAMultitaskFacerecognition1(BaseModel):
    def __init__(self, config=None, device=None, tag='MICAMultitaskFacerecognition1'):
        super(MICAMultitaskFacerecognition1, self).__init__(config, device, tag)
        self.initialize()

    def create_model(self, model_cfg):
        mapping_layers = model_cfg.mapping_layers
        pretrained_path = None
        if not model_cfg.use_pretrained:
            pretrained_path = model_cfg.arcface_pretrained_model
        self.arcface = Arcface(pretrained_path=pretrained_path).to(self.device)
        self.flameModel = Generator(512, 300, self.cfg.model.n_shape, mapping_layers, model_cfg, self.device)



        self.faceClassifier = ArcFace_MLP(num_classes=self.cfg.model.num_classes, margin=self.cfg.train.arcface_margin1, scale=self.cfg.train.arcface_scale1, cfg=self.cfg, device=self.device).to(self.device)

    def load_model(self):
        model_path = os.path.join(self.cfg.output_dir, 'model.tar')
        if os.path.exists(self.cfg.pretrained_model_path) and self.cfg.model.use_pretrained:
            model_path = self.cfg.pretrained_model_path
        if os.path.exists(model_path):
            logger.info(f'[{self.tag}] Trained model found. Path: {model_path} | GPU: {self.device}')
            checkpoint = torch.load(model_path)
            if 'arcface' in checkpoint:
                self.arcface.load_state_dict(checkpoint['arcface'])
            if 'flameModel' in checkpoint:
                self.flameModel.load_state_dict(checkpoint['flameModel'])
            if 'faceClassifier' in checkpoint:
                self.faceClassifier.load_state_dict(checkpoint['faceClassifier'])
        else:
            logger.info(f'[{self.tag}] Checkpoint not available starting from scratch!')

    def model_dict(self):
        return {
            'flameModel': self.flameModel.state_dict(),
            'arcface': self.arcface.state_dict(),
            'faceClassifier': self.faceClassifier.state_dict()
        }

    def parameters_to_optimize(self):
        return [
            {'params': self.flameModel.parameters(), 'lr': self.cfg.train.lr},
            {'params': self.arcface.parameters(), 'lr': self.cfg.train.arcface_lr},
            {'params': self.faceClassifier.parameters(), 'lr': self.cfg.train.face_recog_lr}
        ]

    def encode(self, images, arcface_imgs):
        codedict = {}
        codedict['arcface'] = F.normalize(self.arcface(arcface_imgs))
        codedict['images'] = images
        return codedict

    def decode(self, codedict, epoch=0):
        self.epoch = epoch

        flame_verts_shape = None
        shapecode = None

        if not self.testing:
            flame = codedict['flame']
            shapecode = flame['shape_params'].view(-1, flame['shape_params'].shape[2])
            shapecode = shapecode.to(self.device)[:, :self.cfg.model.n_shape]
            with torch.no_grad():
                flame_verts_shape, _, _ = self.flame(shape_params=shapecode)

        identity_code = codedict['arcface']
        pred_canonical_vertices, pred_shape_code = self.flameModel(identity_code)











        if self.cfg.model.face_embed == 'original-arcface':
            face_embed = identity_code

        elif self.cfg.model.face_embed == 'arcface':
            face_embed = identity_code            # ArcFace embedding (512)


            face_embed = face_embed - face_embed.min(1, keepdim=True)[0]
            face_embed = face_embed / face_embed.max(1, keepdim=True)[0]     

        elif self.cfg.model.face_embed == '3dmm':
            face_embed = pred_shape_code          # 3DMM embedding (300)
            if self.cfg.dataset.norm == '0,1_min-max':


                face_embed = face_embed - face_embed.min(1, keepdim=True)[0]
                face_embed = face_embed / face_embed.max(1, keepdim=True)[0]
            elif self.cfg.dataset.norm == '-1,1_mean-0':

                face_embed = face_embed / 4.0

        elif self.cfg.model.face_embed == 'arcface-3dmm':
            norm_identity_code = identity_code - identity_code.min(1, keepdim=True)[0]

            norm_identity_code = norm_identity_code / norm_identity_code.max(1, keepdim=True)[0]

            norm_pred_shape_code = pred_shape_code - pred_shape_code.min(1, keepdim=True)[0]

            norm_pred_shape_code = norm_pred_shape_code / norm_pred_shape_code.max(1, keepdim=True)[0]

            face_embed = torch.cat((norm_identity_code, norm_pred_shape_code), dim=1)

        elif self.cfg.model.face_embed == 'pc_vertices':
            face_embed = pred_canonical_vertices  # FLAME Point Cloud (5023)

        face_embedd, logits_pred, prob_pred, y_pred = self.faceClassifier(face_embed)
        
        y_true = None
        if 'y_true' in codedict:
            y_true = codedict['y_true']


        output = {
            'flame_verts_shape': flame_verts_shape,
            'flame_shape_code': shapecode,
            'pred_canonical_shape_vertices': pred_canonical_vertices,
            'pred_shape_code': pred_shape_code,
            'faceid': codedict['arcface'],

            'face_embedd': face_embedd,
            'logits_pred': logits_pred,
            'prob_pred': prob_pred,
            'y_pred': y_pred,
            'y_true': y_true
        }

        return output




    def cross_entropy_loss1(self, y_pred, y_true):
        loss = torch.nn.CrossEntropyLoss()
        output = loss(y_pred, y_true)

        return output

    def cross_entropy_loss2(self, logits_pred, y_true):



        output = -torch.mean(torch.sum(F.log_softmax(logits_pred, dim=-1) * y_true, dim=1))
        return output



    def arcface_loss1(self, cosine=None, labels=None):

        mask = labels
        cosine_of_target_classes = cosine[mask == 1] # (None, )

        eps = 1e-6  # theta in the paper
        angles = torch.acos(torch.clamp(cosine_of_target_classes, -1 + eps, 1 - eps))
        modified_cosine_of_target_classes = torch.cos(angles + self.faceClassifier.margin)

        diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(1) # (None,1)
        logits = cosine + (mask * diff) # (None, n_classes)

        logits = logits * self.faceClassifier.scale

        labels = torch.argmax(labels, dim=1)
        return torch.nn.CrossEntropyLoss()(logits, labels)




    def compute_accuracy(self, y_pred, y_true):
        y_true = torch.argmax(y_true, dim=1)




        tp = torch.sum(y_pred == y_true)
        acc = tp / y_pred.size(0)
        return acc




    def compute_losses(self, configs, input, encoder_output, decoder_output):
        losses = {}





        pred_verts = decoder_output['pred_canonical_shape_vertices']
        gt_verts = decoder_output['flame_verts_shape'].detach()

        pred_verts_shape_canonical_diff = (pred_verts - gt_verts).abs()

        if self.use_mask:
            pred_verts_shape_canonical_diff *= self.vertices_mask





        losses['pred_verts_shape_canonical_diff'] = configs.train.lambda1 * torch.mean(pred_verts_shape_canonical_diff) * 1000.0


        logits_pred = decoder_output['logits_pred']
        prob_pred = decoder_output['prob_pred']
        y_pred = decoder_output['y_pred']
        y_true = decoder_output['y_true']












        if configs.train.loss == 'arcface':
            losses['class_loss'] = configs.train.lambda2 * self.arcface_loss1(logits_pred, y_true)
        elif configs.train.loss == 'cross-entropy':
            losses['class_loss'] = configs.train.lambda2 * self.cross_entropy_loss2(logits_pred, y_true)

        metrics = {}
        metrics['acc'] = self.compute_accuracy(y_pred, y_true)

        return losses, metrics
