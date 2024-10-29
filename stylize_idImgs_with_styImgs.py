import os, sys
import numpy as np
import glob
from argparse import ArgumentParser
import shutil
import json
import random
random.seed(440)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--id-imgs', type=str, default='/datasets2/bjgbiesseck/face_recognition/synthetic/tcdiff_with_pretrained_models/tcdiff_original_synthetic_ids/tcdiff_original_10000_synthetic_ids')
    parser.add_argument('--sty-imgs', type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112')
    parser.add_argument('--img-ext', type=str, default='.png')
    parser.add_argument('--file-map-id-sty-imgs', type=str, default='/datasets2/bjgbiesseck/face_recognition/synthetic/tcdiff_with_pretrained_models/tcdiff_original_synthetic_ids/mapping_idImg=tcdiffOriginal10000SyntheticIdsToStyImg_styImg=1CASIAWebFace.json')
    parser.add_argument('--output-path', type=str, default='')
    args = parser.parse_args()
    return args


def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def load_from_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def find_all_files(folder_path, extensions=['.jpg', '.png']):
    image_paths = []
    num_found_files = 0
    for root, _, files in os.walk(folder_path):
        for ext in extensions:
            pattern = os.path.join(root, '*' + ext)
            matching_files = glob.glob(pattern)
            image_paths.extend(matching_files)
            num_found_files += len(matching_files)
            print(f'    num_found_files: {num_found_files}', end='\r')
    print('')
    return sorted(image_paths)


def main(args):
    print('Searching id images')
    id_imgs_paths = find_all_files(args.id_imgs, extensions=[args.img_ext])

    print('Searching style images')
    sty_imgs_paths = find_all_files(args.sty_imgs, extensions=[args.img_ext])


    print(f'Loading dict map: {args.file_map_id_sty_imgs}')
    dict_map_idImgs_styImgs = load_from_json(args.file_map_id_sty_imgs)


    dict_map_idImgs_styImgs = adjust_paths_idImgs_styImgs(dict_map_idImgs_styImgs, id_imgs_paths, sty_imgs_paths)

    if args.output_path == '':
        output_dir = 'stylized='+args.id_imgs.split('/')[-1]+'_styImg='+args.sty_imgs.split('/')[-1]
        args.output_path = os.path.join(os.path.dirname(args.id_imgs), output_dir)
    os.makedirs(args.output_path, exist_ok=True)

    stylize_idImgs_with_styImgs(id_imgs_paths, sty_imgs_paths, args)

    output_file_name = 'mapping_idImg_to_styImg.json'
    output_file_path = os.path.join(args.output_path, output_file_name)

    print(f'Saving dict map: {output_file_path}')
    save_to_json(dict_map_idImgs_styImgs, output_file_path)


    print('\nFinished!\n')


if __name__ == "__main__":
    args = parse_args()
    main(args)