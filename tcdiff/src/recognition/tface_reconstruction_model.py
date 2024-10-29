import argparse
import os, sys
import random
from glob import glob
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import face_align
from loguru import logger
from pytorch3d.io import save_ply, save_obj
from skimage.io import imread
from tqdm import tqdm


from MICA.configs.config import get_cfg_defaults
from MICA.datasets.creation.util import get_arcface_input, get_center
from MICA.utils import util


def get_args_defaults():
    parser = argparse.ArgumentParser(description='MICA - Towards Metrical Reconstruction of Human Faces')
    parser.add_argument('-m', default='data/pretrained/mica.tar', type=str, help='Pretrained model path')
    args = parser.parse_args()
    return args


def load_checkpoint(weights, mica):
    checkpoint = torch.load(weights)
    if 'arcface' in checkpoint:
        mica.arcface.load_state_dict(checkpoint['arcface'])
    if 'flameModel' in checkpoint:
        mica.flameModel.load_state_dict(checkpoint['flameModel'])


def get_MICA(input_size):
    print('getting default MICA configs')
    cfg = get_cfg_defaults()
    mica_weights = 'src/MICA/data/pretrained/mica.tar'

    device = 'cuda:0'
    cfg.model.testing = True
    mica = util.find_model_using_name(model_dir='MICA.micalib.models', model_name=cfg.model.name)(cfg, device)
    print('loading MICA checkpoint:', mica_weights)
    load_checkpoint(mica_weights, mica)
    mica.eval()
    return mica

