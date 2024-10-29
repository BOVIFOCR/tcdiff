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


def main():

    parser = ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='model.ckpt')
    parser.add_argument('--num_image_per_subject', type=int, default=1)
    parser.add_argument('--num_subject', type=int, default=1)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_partition', type=int, default=1)
    parser.add_argument('--partition_idx', type=int, default=0)

    parser.add_argument('--save_root', type=str, default=None)
    parser.add_argument('--id_images_root', default='sample_images/id_images/2377.jpg')
    parser.add_argument('--style_images_root', type=str, default='sample_images/style_images/combined')
    parser.add_argument('--file_map_id_sty_imgs', type=str, default='/datasets2/bjgbiesseck/face_recognition/synthetic/tcdiff_with_pretrained_models/tcdiff_original_synthetic_ids/mapping_idImg=tcdiffOriginal10000SyntheticIdsToStyImg_styImg=1CASIAWebFace.json')

    parser.add_argument('--style_sampling_method', type=str, default='list',
                        choices=['random', 'feature_sim_center:topk_sampling_top1',
                                 'feature_sim_center:top1_sampling_topk', 'list'])
    parser.add_argument('--use_writer', action='store_true')

    parser.add_argument('--start_label', type=int, default=-1)   # -1 = start from first label and first image 

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.id_images_root = os.path.join(root, args.id_images_root)
    args.style_images_root = os.path.join(root, args.style_images_root)

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

    style_images = get_all_files(args.style_images_root, extension_list=['.png', '.jpg', '.jpeg'])
    style_images = natural_sort(style_images)


    style_dataset = ListDatasetWithIndex(style_images, flip_color=True)
    assert len(style_dataset) > 0, args.style_images_root


    if os.path.isdir(args.id_images_root):

        id_images = get_all_files(args.id_images_root, extension_list=['.png', '.jpg', '.jpeg'], sorted=True)


        if len(id_images) < args.num_subject:
            id_images = id_images * args.num_subject
        id_dataset = ListDatasetWithIndex(id_images, flip_color=True)
    elif os.path.isfile(args.id_images_root):
        id_images = [args.id_images_root] * len(style_dataset)
        args.num_image_per_subject = len(style_dataset)
        id_dataset = ListDatasetWithIndex(id_images, flip_color=True)
    else:
        print('id_images_root is not a file or directory')
        raise ValueError(args.id_images_root)
    





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
                     num_image_per_subject=args.num_image_per_subject, num_subject=args.num_subject,
                     batch_size=args.batch_size, num_workers=args.num_workers, save_root=args.save_root,
                     style_sampling_method=args.style_sampling_method,
                     num_partition=args.num_partition, partition_idx=args.partition_idx, writer=writer, start_label=args.start_label)

    if args.use_writer:
        writer.close()

if __name__ == "__main__":
    main()
