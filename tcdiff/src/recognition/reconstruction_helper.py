import subprocess, sys, os
sys.path.append(os.getcwd().split('datagen_framework')[0] + 'datagen_framework')
import os
import torch
from torch import nn as nn
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
from src.recognition.tface_model import Backbone
from src.recognition.adaface import AdaFaceV3
from src.general_utils import os_utils

from src.recognition import tface_reconstruction_model
from functools import partial
from typing import Dict


def disabled_train(mode=True, self=None):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def same_config(config1, config2, skip_keys=[]):
    for key in config1.keys():
        if key in skip_keys:
            pass
        else:
            if config1[key] != config2[key]:
                return False
    return True


def download_ir_pretrained_statedict(backbone_name, dataset_name, loss_fn):

    if backbone_name == 'ir_101' and dataset_name == 'webface4m' and loss_fn == 'adaface':
        root = os_utils.get_project_root(project_name='tcdiff')
        _name, _id = 'adaface_ir101_webface4m.ckpt', '18jQkqB0avFqWa0Pas52g54xNshUOQJpQ'
    elif backbone_name == 'ir_50' and dataset_name == 'webface4m' and loss_fn == 'adaface':
        root = os_utils.get_project_root(project_name='tcdiff')
        _name, _id = 'adaface_ir50_webface4m.ckpt', '1BmDRrhPsHSbXcWZoYFPJg2KJn1sd3QpN'
    else:
        raise NotImplementedError()
    checkpoint_path = os.path.join(root, 'pretrained_models', _name)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    if not os.path.isfile(checkpoint_path):
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'gdown'])
        try:
            subprocess.check_call([os.path.expanduser('~/.local/bin/gdown'), '--id', _id])
        except:

            subprocess.check_call([os.path.expanduser('gdown'), '--id', _id])
        if not os.path.isdir(os.path.dirname(checkpoint_path)):
            subprocess.check_call(['mkdir', '-p', os.path.dirname(checkpoint_path)])
        subprocess.check_call(['mv', _name, checkpoint_path])

    assert os.path.isfile(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    model_statedict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
    return model_statedict


def get_dim(style_dims=[]):
    cdim = 0
    for index in style_dims:
        if index == 2:
            cdim += 64
        if index == 4:
            cdim += 128
        if index == 6:
            cdim += 256
        if index == 8:
            cdim += 512
    return cdim

def get_spatial(style_dims=[]):
    spatial_dim = []
    for index in style_dims:
        if index == 2:
            spatial_dim.append((56,56))
        if index == 4:
            spatial_dim.append((28,28))
        if index == 6:
            spatial_dim.append((14,14))
        if index == 8:
            spatial_dim.append((7,7))
    return spatial_dim

dict_name_to_filter = {
    "PIL": {
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
        "lanczos": Image.LANCZOS,
        "box": Image.BOX
    }
}


def resize_images(x, resizer, ToTensor, mean, std, device):
    x = x.transpose((0, 2, 3, 1))
    x = list(map(lambda x: ToTensor(resizer(x)), list(x)))
    x = torch.stack(x, 0).to(device)
    x = (x/255.0 - mean)/std
    return x


def make_resizer(library, filter, output_size):
    if library == "PIL":
        s1, s2 = output_size
        def resize_single_channel(x_np):
            img = Image.fromarray(x_np.astype(np.float32), mode='F')
            img = img.resize(output_size, resample=dict_name_to_filter[library][filter])
            return np.asarray(img).reshape(s1, s2, 1)
        def func(x):
            x = [resize_single_channel(x[:, :, idx]) for idx in range(3)]
            x = np.concatenate(x, axis=2).astype(np.float32)
            return x
    elif library == "PyTorch":
        import warnings

        warnings.filterwarnings("ignore")
        def func(x):
            x = torch.Tensor(x.transpose((2, 0, 1)))[None, ...]
            x = F.interpolate(x, size=output_size, mode=filter, align_corners=False)
            x = x[0, ...].cpu().data.numpy().transpose((1, 2, 0)).clip(0, 255)
            return x
    else:
        raise NotImplementedError('library [%s] is not include' % library)
    return func



def return_head(head_name='adaface', class_num=205990, head_m=0.4):
    if head_name == 'adaface':
        head = AdaFaceV3(embedding_size=512,
                         classnum=class_num,
                         m=head_m,
                         scaler_fn='batchnorm',
                         rad_h=-0.333,
                         s=64.0,
                         t_alpha=0.01,
                         cut_gradient=True,
                         head_b=0.4)
    elif head_name == '' or head_name == 'none':
        return None
    else:
        raise ValueError('not implemented yet')
    return head





class ReconstructionModel(nn.Module):

    def __init__(self, backbone, reconstruction_config):
        super(ReconstructionModel, self).__init__()
        self.backbone = backbone
        self.reconstruction_config = reconstruction_config

        self.size = 112
        self.resizer = make_resizer("PIL", "bilinear", (self.size, self.size))
        self.totensor = transforms.ToTensor()
        self.mean = torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        self.std = torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        self.swap_channel = True

    def get_arcface_embedding(self, x):
        embedd = F.normalize(self.backbone.arcface(x))
        return embedd

    def forward(self, embedd):
        pred_pointcloud, pred_3dmm = self.backbone.flameModel(embedd)
        render_image = None
        return pred_pointcloud, pred_3dmm, render_image
        


def make_3d_face_reconstruction_model(reconstruction_config, enable_training=False):

    if not reconstruction_config:
        return None

    if 'MICA' == reconstruction_config.backbone:
        print('\nmaking MICA')
        backbone = tface_reconstruction_model.get_MICA(input_size=(112, 112))

    else:
        raise NotImplementedError()

    model = ReconstructionModel(backbone=backbone, reconstruction_config=reconstruction_config)
    if enable_training:
        print('enable training')
        pass
    else:
        model = model.eval()
        model.train = partial(disabled_train, self=model)
        for param in model.parameters():
            param.requires_grad = False

    return model