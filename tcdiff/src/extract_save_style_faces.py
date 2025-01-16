import pandas as pd
import pyrootutils
import dotenv
import os, sys
import json
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
from PIL import Image

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
dotenv.load_dotenv(dotenv_path=root.parent.parent / '.env', override=True)
assert os.getenv('DATA_ROOT')

LOG_ROOT = str(root.parent.parent / 'experiments')
os.environ.update({'LOG_ROOT': LOG_ROOT})
os.environ.update({'PROJECT_TASK': root.stem})
os.environ.update({'REPO_ROOT': str(root.parent.parent)})

import hydra
from pytorch_lightning import LightningModule, LightningDataModule
from argparse import ArgumentParser
import omegaconf
from src.visualizations.extra_visualization import dataset_generate
from src.general_utils.os_utils import get_all_files
from src.visualizations.extra_visualization import ListDatasetWithIndex
import numpy as np
from src.visualizations.record import Writer
import re

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def get_all_files_in_path(folder_path, file_extension=['.jpg','.jpeg','.png'], pattern=''):
    file_list = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            path_file = os.path.join(root, filename)
            for ext in file_extension:
                if pattern in path_file and path_file.lower().endswith(ext.lower()):
                    file_list.append(path_file)
    # file_list.sort()
    file_list = natural_sort(file_list)
    return file_list

# based on class ListDatasetWithIndex (tcdiff/src/visualizations/extra_visualization.py)
def load_normalize_img(img_path='/path/to/img.jpg', flip_color=True):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(img_path)
    img = img[:, :, :3]
    if flip_color:
        img = img[:, :, ::-1]
    img = Image.fromarray(img)
    img = transform(img)
    return img

def get_face_style_embedding(args, pl_module, img_normalized):
    generator = torch.manual_seed(args.seed)

    if len(img_normalized.shape) == 3:
        img_normalized = torch.unsqueeze(img_normalized, dim=0)
    img_normalized = img_normalized.to(pl_module.device)
    # print('img_normalized.shape:', img_normalized.shape)
    # print('img_normalized.device:', img_normalized.device)

    id_feat, id_cross_att = pl_module.label_mapping(img_normalized)
    _, spatial = pl_module.recognition_model(img_normalized.to(pl_module.device))
    ext_mapping = pl_module.external_mapping(spatial)
    if type(spatial) == list: spatial = spatial[0]

    x = spatial
    B,C,H,W = x.shape
    mean  = x.view(B,C,-1).mean(-1, keepdim=True)
    std   = x.view(B,C,-1).std(-1, keepdim=True)
    style = torch.cat([mean, std], dim=-1)

    # print('ORIGINAL:')
    # print('id_feat.shape:', id_feat.shape)
    # print('id_cross_att.shape:', id_cross_att.shape)
    # print('spatial.shape:', spatial.shape)
    # print('ext_mapping.shape:', ext_mapping.shape)
    # print('style.shape:', style.shape)
    # print('--------')

    # id_feat      = torch.flatten(id_feat, start_dim=1)
    # id_cross_att = torch.flatten(id_cross_att, start_dim=1)
    # spatial      = torch.flatten(spatial, start_dim=1)
    # ext_mapping  = torch.flatten(ext_mapping, start_dim=1)
    # style        = torch.flatten(style, start_dim=1)
    # print('FLATTEN:')
    # print('id_feat.shape:', id_feat.shape)
    # print('id_cross_att.shape:', id_cross_att.shape)
    # print('spatial.shape:', spatial.shape)
    # print('ext_mapping.shape:', ext_mapping.shape)
    # print('style.shape:', style.shape)
    # print('--------')

    return id_feat, id_cross_att, spatial, ext_mapping, style


def main():
    parser = ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='model.ckpt')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--imgs', type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112')
    parser.add_argument('--start-idx', type=int, default=0)

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f'\nLoading checkpoint: \'{args.ckpt_path}\'')
    ckpt = torch.load(os.path.join(root, args.ckpt_path))
    model_hparam = ckpt['hyper_parameters']
    model_hparam['unet_config']['params']['pretrained_model_path'] = os.path.join(root, model_hparam['unet_config']['params']['pretrained_model_path'])
    model_hparam['recognition']['ckpt_path'] = os.path.join(root, model_hparam['recognition']['ckpt_path'])
    model_hparam['recognition']['center_path'] = os.path.join(root, model_hparam['recognition']['center_path'])
    model_hparam['recognition_eval']['center_path'] = os.path.join(root, model_hparam['recognition_eval']['center_path'])

    model_hparam['_target_'] = 'src.trainer.Trainer'
    model_hparam['_partial_'] = True

    print('')
    pl_module: LightningModule = hydra.utils.instantiate(model_hparam)()
    print('Instantiated ', model_hparam['_target_'])

    print(f'Setting loaded weights...\n')
    pl_module.load_state_dict(ckpt['state_dict'], strict=True)
    pl_module.to('cuda')
    pl_module.eval()

    args.imgs = args.imgs.rstrip('/')
    output_path = args.imgs.rstrip('/') + '_STYLE_FEATURES'
    os.makedirs(output_path, exist_ok=True)

    print(f'Searching images in \'{args.imgs}\'')
    imgs_paths = get_all_files_in_path(args.imgs)
    print(f'Found {len(imgs_paths)} images\n------------------\n')

    total_elapsed_time = 0.0
    for idx_path, path_img in enumerate(imgs_paths):
        if idx_path >= args.start_idx:
            start_time = time.time()
            print(f'{idx_path}/{len(imgs_paths)} - Computing style features')
            print(f'path_img: {path_img}')
            img = load_normalize_img(path_img)
            id_feat_img, id_cross_att_img, spatial_img, ext_mapping_img, style_img = get_face_style_embedding(args, pl_module, img)
            
            output_path_dir = os.path.dirname(path_img.replace(args.imgs, output_path))
            print(f'output_path_dir: {output_path_dir}')
            os.makedirs(output_path_dir, exist_ok=True)

            img_name, img_ext = os.path.splitext(os.path.basename(path_img))
            output_path_id_feat      = os.path.join(output_path_dir, img_name+'_id_feat.pt')
            output_path_id_cross_att = os.path.join(output_path_dir, img_name+'_id_cross_att.pt')
            output_path_spatial      = os.path.join(output_path_dir, img_name+'_spatial.pt')
            output_path_ext_mapping  = os.path.join(output_path_dir, img_name+'_ext_mapping.pt')
            output_path_style        = os.path.join(output_path_dir, img_name+'_style.pt')
            print('output_path_id_feat:', output_path_id_feat)
            torch.save(id_feat_img, output_path_id_feat)
            print('output_path_id_cross_att:', output_path_id_cross_att)
            torch.save(id_cross_att_img, output_path_id_cross_att)
            print('output_path_spatial:', output_path_spatial)
            torch.save(spatial_img, output_path_spatial)
            print('output_path_ext_mapping:', output_path_ext_mapping)
            torch.save(ext_mapping_img, output_path_ext_mapping)
            print('output_path_style:', output_path_style)
            torch.save(style_img, output_path_style)

            elapsed_time = time.time()-start_time
            total_elapsed_time += elapsed_time
            avg_sample_time = total_elapsed_time / ((idx_path-args.start_idx)+1)
            estimated_time = avg_sample_time * (len(imgs_paths)-(idx_path+1))
            print("Elapsed time: %.3fs" % elapsed_time)
            print("Avg elapsed time: %.3fs" % avg_sample_time)
            print("Total elapsed time: %.3fs,  %.3fm,  %.3fh" % (total_elapsed_time, total_elapsed_time/60, total_elapsed_time/3600))
            print("Estimated Time to Completion (ETC): %.3fs,  %.3fm,  %.3fh" % (estimated_time, estimated_time/60, estimated_time/3600))
            print('--------------')

        else:
            print(f'Skipping indices: {idx_path}/{len(imgs_paths)}', end='\r')

    print('\nFinished!')


if __name__ == "__main__":
    main()