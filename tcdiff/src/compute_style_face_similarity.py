import pandas as pd
import pyrootutils
import dotenv
import os, sys
import json
import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

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

def get_all_files_in_path(folder_path, file_extension=['.jpg','.png'], pattern=''):
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

def load_from_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def adjust_paths_idImgs_styImgs(dict_map_idImgs_styImgs, id_imgs_paths, sty_imgs_paths):
    pass


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
    mean = x.view(B,C,-1).mean(-1, keepdim=True)
    std = x.view(B,C,-1).std(-1, keepdim=True)
    style = torch.cat([mean, std], dim=-1)

    # print('ORIGINAL:')
    # print('id_feat.shape:', id_feat.shape)
    # print('id_cross_att.shape:', id_cross_att.shape)
    # print('spatial.shape:', spatial.shape)
    # print('ext_mapping.shape:', ext_mapping.shape)
    # print('style.shape:', style.shape)
    # print('--------')

    id_feat = torch.flatten(id_feat, start_dim=1)
    id_cross_att = torch.flatten(id_cross_att, start_dim=1)
    spatial = torch.flatten(spatial, start_dim=1)
    ext_mapping = torch.flatten(ext_mapping, start_dim=1)
    style = torch.flatten(style, start_dim=1)
    # print('FLATTEN:')
    # print('id_feat.shape:', id_feat.shape)
    # print('id_cross_att.shape:', id_cross_att.shape)
    # print('spatial.shape:', spatial.shape)
    # print('ext_mapping.shape:', ext_mapping.shape)
    # print('style.shape:', style.shape)
    # print('--------')

    return id_feat, id_cross_att, spatial, ext_mapping, style


def save_heatmap_with_images(list1_images, list2_images, similarity_matrix, title, output_path):
    if type(list1_images) is list and type(list1_images[0]) is str:
        for idx_img1, path_img1 in enumerate(list1_images):
            list1_images[idx_img1] = Image.open(path_img1)
        for idx_img2, path_img2 in enumerate(list2_images):
            list2_images[idx_img2] = Image.open(path_img2)

    similarity_matrix = np.array(similarity_matrix)
    assert similarity_matrix.shape[0] == similarity_matrix.shape[1], "Matrix must be square."

    # Create the heatmap
    # figsize = 8   # good for 6x6 images
    # figsize = 11  # good for 8x8 images
    figsize = (((len(list1_images)+1)*240) + 540) / 246
    fig, ax = plt.subplots(figsize=(figsize, figsize))  # Adjust size to fit images better
    cax = ax.imshow(similarity_matrix, cmap="gist_gray", aspect="equal", vmin=0.0, vmax=1.0)

    ax.xaxis.tick_top()

    plt.rcParams['axes.titlepad'] = 70
    plt.title(title)
    fig.colorbar(cax, ax=ax, orientation="vertical")

    def add_image_labels(images, axis, is_x_axis=True):
        for i, img in enumerate(images):
            imagebox = OffsetImage(img, zoom=0.5)  # Adjust zoom to control image size
            if is_x_axis:
                ab = AnnotationBbox(imagebox, (i, -0.5), frameon=False, box_alignment=(0.5, 0))
                axis.add_artist(ab)
            else:
                ab = AnnotationBbox(imagebox, (-0.5, i), frameon=False, box_alignment=(1, 0.5))
                axis.add_artist(ab)

    add_image_labels(list1_images, ax, is_x_axis=True)
    add_image_labels(list2_images, ax, is_x_axis=False)
    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            ax.text(j, i, f"{similarity_matrix[i, j]:.2f}", ha="center", va="center", color="white" if similarity_matrix[i, j] < 0.5 else "black")
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1))

    ax.set_xticks(range(len(list1_images)))
    ax.set_yticks(range(len(list2_images)))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)



def main():
    parser = ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='model.ckpt')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--img1', type=str, default='/path/to/some/img.png')
    parser.add_argument('--img2', type=str, default='/path/to/some/img.png')

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

    print(f'\nSetting loaded weights...')
    pl_module.load_state_dict(ckpt['state_dict'], strict=True)
    pl_module.to('cuda')
    pl_module.eval()


    # only 2 images
    if os.path.isfile(args.img1) and os.path.isfile(args.img2):
        print(f'Loading \'{args.img1}\'')
        img1 = load_normalize_img(args.img1)
        print(f'Loading \'{args.img2}\'')
        img2 = load_normalize_img(args.img2)

        id_feat_img1, id_cross_att_img1, spatial_img1, ext_mapping_img1, style_img1 = get_face_style_embedding(args, pl_module, img1)
        id_feat_img2, id_cross_att_img2, spatial_img2, ext_mapping_img2, style_img2 = get_face_style_embedding(args, pl_module, img2)
        
        spatial_euclidist = torch.cdist(spatial_img1, spatial_img2)
        spatial_cossim = F.cosine_similarity(spatial_img1, spatial_img2)

        print('spatial_euclidist:', spatial_euclidist)
        print('spatial_cossim:', spatial_cossim)
    
    # folder of N images
    elif os.path.isdir(args.img1) and os.path.isdir(args.img2):
        print(f'Loading images from \'{args.img1}\'')
        img1_paths = get_all_files_in_path(args.img1)
        # print('img1_paths:', img1_paths)
        img2_paths = get_all_files_in_path(args.img2)

        # get outputs dimensions
        img1 = load_normalize_img(img1_paths[0])
        id_feat_img1, id_cross_att_img1, spatial_img1, ext_mapping_img1, style_img1 = get_face_style_embedding(args, pl_module, img1)
        id_feat_imgs1 = torch.zeros((len(img1_paths),512))
        id_feat_imgs2 = torch.zeros((len(img2_paths),512))
        spatial_dim = spatial_img1.shape[1]
        spatial_imgs1 = torch.zeros((len(img1_paths),spatial_dim))
        spatial_imgs2 = torch.zeros((len(img2_paths),spatial_dim))
        style_dim = style_img1.shape[1]
        style_imgs1 = torch.zeros((len(img1_paths),style_dim))
        style_imgs2 = torch.zeros((len(img2_paths),style_dim))

        for idx_img1, path_img1 in enumerate(img1_paths):
            print(f'{idx_img1}/{len(img1_paths)-1} - Computing style features imgs1', end='\r')
            img1 = load_normalize_img(path_img1)
            id_feat_img1, id_cross_att_img1, spatial_img1, ext_mapping_img1, style_img1 = get_face_style_embedding(args, pl_module, img1)
            id_feat_imgs1[idx_img1] = id_feat_img1
            spatial_imgs1[idx_img1] = spatial_img1
            style_imgs1[idx_img1] = style_img1
        print()

        for idx_img2, path_img2 in enumerate(img2_paths):
            print(f'{idx_img1}/{len(img1_paths)-1} - Computing style features  imgs2', end='\r')
            img2 = load_normalize_img(path_img2)
            id_feat_img2, id_cross_att_img2, spatial_img2, ext_mapping_img2, style_img2 = get_face_style_embedding(args, pl_module, img2)
            id_feat_imgs2[idx_img2] = id_feat_img2
            spatial_imgs2[idx_img2] = spatial_img2
            style_imgs2[idx_img2] = style_img2
        print('\n')

        id_feats_cossims   = torch.zeros((len(img1_paths),len(img2_paths)), dtype=float)
        spatial_euclidists = torch.zeros((len(img1_paths),len(img2_paths)), dtype=float)
        spatial_cossims    = torch.zeros((len(img1_paths),len(img2_paths)), dtype=float)
        style_cossims      = torch.zeros((len(img1_paths),len(img2_paths)), dtype=float)
        for idx_img1, spatial_img1 in enumerate(spatial_imgs1):
            id_feat_img1 = F.normalize(torch.unsqueeze(id_feat_imgs1[idx_img1,:], dim=0))
            spatial_img1 = F.normalize(torch.unsqueeze(spatial_img1, dim=0))
            style_img1   = F.normalize(torch.unsqueeze(style_imgs1[idx_img1,:], dim=0))
            for idx_img2, spatial_img2 in enumerate(spatial_imgs2):
                id_feat_img2 = F.normalize(torch.unsqueeze(id_feat_imgs2[idx_img2,:], dim=0))
                spatial_img2 = F.normalize(torch.unsqueeze(spatial_img2, dim=0))
                style_img2   = F.normalize(torch.unsqueeze(style_imgs2[idx_img2,:], dim=0))

                id_feats_cossims[idx_img1,idx_img2]   = F.cosine_similarity(id_feat_img1, id_feat_img2)
                spatial_euclidists[idx_img1,idx_img2] = torch.cdist(spatial_img1, spatial_img2)
                spatial_cossims[idx_img1,idx_img2]    = F.cosine_similarity(spatial_img1, spatial_img2)
                style_cossims[idx_img1,idx_img2]      = torch.cdist(style_img1, style_img2)

        print('spatial_euclidists:\n', spatial_euclidists)
        print('spatial_cossims:\n', spatial_cossims)
        print('style_cossims:\n', style_cossims)
        print('id_feats_cossims:\n', id_feats_cossims)


        title = 'Cosine Similarity (STYLE)'
        path_dist_matrix = f"{'-'.join(args.img1.split('/')[-2:])}_{'-'.join(args.img2.split('/')[-2:])}_{len(img1_paths)}x{len(img2_paths)}_style_similarity_matrix.png"
        print(f'Saving similarity matrix: {path_dist_matrix}')
        save_heatmap_with_images(img1_paths, img2_paths, spatial_cossims.numpy(), title, path_dist_matrix)

        title = 'Euclidean Distances (STYLE MEAN/STD)'
        path_dist_matrix = f"{'-'.join(args.img1.split('/')[-2:])}_{'-'.join(args.img2.split('/')[-2:])}_{len(img1_paths)}x{len(img2_paths)}_style_mean-std_euclidistances_matrix.png"
        print(f'Saving euclidean distances: {path_dist_matrix}')
        save_heatmap_with_images(img1_paths, img2_paths, style_cossims.numpy(), title, path_dist_matrix)

        title = 'Cosine Similarity (ID)'
        path_dist_matrix = f"{'-'.join(args.img1.split('/')[-2:])}_{'-'.join(args.img2.split('/')[-2:])}_{len(img1_paths)}x{len(img2_paths)}_id_similarity_matrix.png"
        print(f'Saving similarity matrix: {path_dist_matrix}')
        save_heatmap_with_images(img1_paths, img2_paths, id_feats_cossims.detach().numpy(), title, path_dist_matrix)


    else:
        if not os.path.isdir(args.img1): print(f'\nNo such directory: \'{args.img1}\'')
        if not os.path.isdir(args.img2): print(f'\nNo such directory: \'{args.img2}\'')


if __name__ == "__main__":
    main()
