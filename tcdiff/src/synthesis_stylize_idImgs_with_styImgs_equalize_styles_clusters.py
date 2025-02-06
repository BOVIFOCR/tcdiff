# duo
# export CUDA_VISIBLE_DEVICES=1; python synthesis_stylize_idImgs_with_styImgs_equalize_styles_clusters.py --ckpt_path /home/bjgbiesseck/GitHub/BOVIFOCR_dcface_synthetic_face/experiments_WITH_BFM_CONSISTENCY_CONSTRAINTS/dcface/e:10_spatial_dim:5_bias:0.0_casia_ir50_09-10_1/checkpoints/epoch_008.ckpt --save_root /datasets2/bjgbiesseck/face_recognition/dcface/generated_images/tcdiff_WITH_BFM_e:10_spatial_dim:5_bias:0.0_casia_ir50_09-10_1_EQUALIZED-STYLES_NCLUSTERS=100 --file_map_id_sty_imgs /datasets2/bjgbiesseck/face_recognition/synthetic/dcface_with_pretrained_models/dcface_original_synthetic_ids/dcface_original_10000_synthetic_ids_STYLE_FEATURES_CLUSTERING_FROM_1_CASIA-WebFace-imgs_crops_112x112_STYLE_FEATURES_CLUSTERING-feature=_style-_distance=cosine-nclusters=100/feature=_style/_distance=cosine/nclusters=100/face_pairs_subj_style.json --batch_size 49
# export CUDA_VISIBLE_DEVICES=1; python synthesis_stylize_idImgs_with_styImgs_equalize_styles_clusters.py --ckpt_path /home/bjgbiesseck/GitHub/BOVIFOCR_dcface_synthetic_face/experiments_WITH_BFM_CONSISTENCY_CONSTRAINTS/dcface/e:10_spatial_dim:5_bias:0.0_casia_ir50_09-10_1/checkpoints/epoch_008.ckpt --save_root /datasets2/bjgbiesseck/face_recognition/dcface/generated_images/tcdiff_WITH_BFM_e:10_spatial_dim:5_bias:0.0_casia_ir50_09-10_1_EQUALIZED-STYLES_NCLUSTERS=100 --file_map_id_sty_imgs /datasets2/bjgbiesseck/face_recognition/synthetic/dcface_with_pretrained_models/dcface_original_synthetic_ids/dcface_original_10000_synthetic_ids_STYLE_FEATURES_CLUSTERING_FROM_1_CASIA-WebFace-imgs_crops_112x112_STYLE_FEATURES_CLUSTERING-feature=_style-_distance=cosine-nclusters=100/feature=_style/_distance=cosine/nclusters=100/face_pairs_subj_style.json --id_images_root /datasets2/bjgbiesseck/face_recognition/synthetic/dcface_with_pretrained_models/dcface_original_synthetic_ids --style_images_root /datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace --batch_size 49

# diolkos
# export CUDA_VISIBLE_DEVICES=0; python synthesis_stylize_idImgs_with_styImgs_equalize_styles_clusters.py --ckpt_path /home/bjgbiesseck/GitHub/BOVIFOCR_dcface_synthetic_face/experiments_WITH_BFM_CONSISTENCY_CONSTRAINTS/dcface/e:10_spatial_dim:5_bias:0.0_casia_ir50_09-10_1/checkpoints/epoch_008.ckpt --save_root /nobackup/unico/datasets/face_recognition/dcface/generated_images/tcdiff_WITH_BFM_e:10_spatial_dim:5_bias:0.0_casia_ir50_09-10_1_EQUALIZED-STYLES_NCLUSTERS=100 --file_map_id_sty_imgs /nobackup/unico/datasets/face_recognition/synthetic/dcface_with_pretrained_models/dcface_original_synthetic_ids/dcface_original_10000_synthetic_ids_STYLE_FEATURES_CLUSTERING_FROM_1_CASIA-WebFace-imgs_crops_112x112_STYLE_FEATURES_CLUSTERING-feature=_style-_distance=cosine-nclusters=100/feature=_style/_distance=cosine/nclusters=100/face_pairs_subj_style.json --id_images_root /nobackup/unico/datasets/face_recognition/synthetic/dcface_with_pretrained_models/dcface_original_synthetic_ids --style_images_root /nobackup/unico/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace --batch_size 49


import pandas as pd
import pyrootutils
import dotenv
import os, sys
import json
import torch
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


def load_from_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def adjust_paths_idImgs_styImgs(dict_map_idImgs_styImgs, id_imgs_paths, sty_imgs_paths):
    pass


# path1       = '/datasets/dataset_name/0/0.jpg'
# path2       = '/home/user/images/dataset_name'
# path_prefix = '/datasets/dataset_name'
def find_path_prefix(path1='', path2=''):
    path1_items = path1.split('/')
    path2_items = path2.split('/')
    if path2_items[-1] in path1_items:
        found_index = path1_items.index(path2_items[-1])
        path_prefix = '/'.join(path1_items[:found_index+1])
    else:
        path_prefix = '/'.join(path1_items[:-2])
    return path_prefix




def main():

    parser = ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='model.ckpt')
    parser.add_argument('--save_root', type=str, default=None)
    parser.add_argument('--file_map_id_sty_imgs', type=str, default='/datasets2/bjgbiesseck/face_recognition/synthetic/dcface_with_pretrained_models/dcface_original_synthetic_ids/dcface_original_10000_synthetic_ids_STYLE_FEATURES_CLUSTERING_FROM_1_CASIA-WebFace-imgs_crops_112x112_STYLE_FEATURES_CLUSTERING-feature=_style-_distance=cosine-nclusters=100/feature=_style/_distance=cosine/nclusters=100/face_pairs_subj_style.json')
    parser.add_argument('--batch_size', type=int, default=16)
    
    parser.add_argument('--dont_save_id_img', action='store_true')
    parser.add_argument('--save_style_img', action='store_true')
    
    # parser.add_argument('--num_image_per_subject', type=int, default=1)
    # parser.add_argument('--num_subject', type=int, default=1)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--num_workers', type=int, default=0)
    
    parser.add_argument('--id_images_root', type=str, default='')      # sample_images/id_images/2377.jpg  or  sample_images/id_images
    parser.add_argument('--style_images_root', type=str, default='')   # sample_images/style_images/combined
    
    parser.add_argument('--style_sampling_method', type=str, default='mapping', choices=['mapping'])
    parser.add_argument('--use_writer', action='store_true')

    parser.add_argument('--num_partition', type=int, default=1)
    parser.add_argument('--partition_idx', type=int, default=0)
    parser.add_argument('--start_label', type=int, default=-1)   # -1 = start from first label and first image 

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # args.id_images_root = os.path.join(root, args.id_images_root)
    # args.style_images_root = os.path.join(root, args.style_images_root)

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


    


    print(f"\nLoading id and style face pairs mapping: '{args.file_map_id_sty_imgs}'")
    all_face_pairs_subj_style = load_from_json(args.file_map_id_sty_imgs)
    
    id_images = []
    style_images = []
    for idx_key, key in enumerate(list(all_face_pairs_subj_style.keys())):
        id_images.append(key)
        style_images.extend(all_face_pairs_subj_style[key])
    num_subject = len(id_images)
    num_image_per_subject = len(all_face_pairs_subj_style[id_images[0]])

    if args.id_images_root != '':
        id_images_path_prefix = find_path_prefix(id_images[0], args.id_images_root)
        print('id_images_path_prefix:', id_images_path_prefix)
        for idx_id_image_path, id_image_path in enumerate(id_images):
            id_images[idx_id_image_path] = id_image_path.replace(id_images_path_prefix, args.id_images_root)

    if args.style_images_root != '':
        style_images_path_prefix = find_path_prefix(style_images[0], args.style_images_root)
        print('style_images_path_prefix:', style_images_path_prefix)
        for idx_style_image_path, style_image_path in enumerate(style_images):
            style_images[idx_style_image_path] = style_image_path.replace(style_images_path_prefix, args.style_images_root)

    id_dataset = ListDatasetWithIndex(id_images, flip_color=True)
    style_dataset = ListDatasetWithIndex(style_images, flip_color=True)





    if args.save_root is None:
        runname_name = os.path.basename(args.ckpt_path).split('.')[0]
        id_name = os.path.basename(args.id_images_root).split('.')[0]
        style_dir_name = os.path.basename(args.style_images_root)
        args.save_root = os.path.join(root, 'generated_images', runname_name, f'id:{id_name}/sty:{args.style_sampling_method}_{style_dir_name}')
        os.makedirs(args.save_root, exist_ok=True)

    print('saving at {}'.format(args.save_root))
    if args.use_writer:
        if args.num_partition > 1:
            args.save_root = os.path.join(args.save_root, f"record_{args.partition_idx}-{args.num_partition}")
        else:
            args.save_root = os.path.join(args.save_root, 'record')
        print('using writer', args.save_root)
        writer = Writer(args.save_root)
    else:
        writer = None

    dataset_generate(pl_module, style_dataset, id_dataset,
                     num_image_per_subject=num_image_per_subject, num_subject=num_subject,
                     batch_size=args.batch_size, num_workers=args.num_workers, save_root=args.save_root,
                     style_sampling_method=args.style_sampling_method,
                     num_partition=args.num_partition, partition_idx=args.partition_idx, writer=writer, start_label=args.start_label, seed=args.seed,
                     save_id_img=not args.dont_save_id_img, save_style_img=args.save_style_img)

    if args.use_writer:
        writer.close()

if __name__ == "__main__":
    main()
