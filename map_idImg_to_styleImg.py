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
    parser.add_argument('--samples-per-id', type=int, default=50)
    parser.add_argument('--output-path', type=str, default='')
    parser.add_argument('--shuffle-sty-imgs', action='store_true')
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


def map_idImgs_styImgs(idImgs_paths, styImgs_paths, args):
    dict_map_idImgs_styImgs = {}
    idx_styImg_path = 0
    for idx_idImg_path, idImg_path in enumerate(idImgs_paths):
        idImg_sub_path = '/'.join(idImg_path.split('/')[-2:])

        
        styImgs_list = []
        for i in range(args.samples_per_id):
            if idx_styImg_path >= len(styImgs_paths):
                idx_styImg_path = 0
            print(f'    idx_idImg_path {idx_idImg_path}/{len(idImgs_paths)} - styImg {i}/{args.samples_per_id}', end='\r')
            styImg_sub_path = '/'.join(styImgs_paths[idx_styImg_path].split('/')[-2:])

            styImgs_list.append(styImg_sub_path)
            idx_styImg_path += 1
        
        dict_map_idImgs_styImgs[idImg_sub_path] = styImgs_list


    print('')
    return dict_map_idImgs_styImgs


def main(args):
    if args.output_path == '':
        args.output_path = os.path.dirname(args.id_imgs)
    os.makedirs(args.output_path, exist_ok=True)

    print('Searching id images')
    id_imgs_paths = find_all_files(args.id_imgs, extensions=[args.img_ext])

    print('Searching style images')
    sty_imgs_paths = find_all_files(args.sty_imgs, extensions=[args.img_ext])


    if args.shuffle_sty_imgs:
        random.shuffle(sty_imgs_paths)

    print('Mapping id imgs to style imgs')
    dict_map_idImgs_styImgs = map_idImgs_styImgs(id_imgs_paths, sty_imgs_paths, args)

    output_file_name = 'mapping_idImg_to_styImg.json'
    output_file_path = os.path.join(args.output_path, output_file_name)





    print(f'Saving dict map: {output_file_path}')
    save_to_json(dict_map_idImgs_styImgs, output_file_path)


    print('\nFinished!\n')


if __name__ == "__main__":
    args = parse_args()
    main(args)