# duo
# export CUDA_VISIBLE_DEVICES=1; python make_face_pairs_dataset_id_style.py --subj-clusters /datasets2/bjgbiesseck/face_recognition/synthetic/dcface_with_pretrained_models/dcface_original_synthetic_ids/dcface_original_10000_synthetic_ids_STYLE_FEATURES_CLUSTERING_FROM_1_CASIA-WebFace-imgs_crops_112x112_STYLE_FEATURES_CLUSTERING-feature=_style-_distance=cosine-nclusters=100/feature=_style/_distance=cosine/nclusters=100/clusters-data_feature=_style.pt_distance=cosine_nclusters=100.pkl --subj-ext .png --num-samples-per-id 49 --style-clusters /datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112_STYLE_FEATURES_CLUSTERING/feature=_style/_distance=cosine/nclusters=100/clusters-data_feature=_style.pt_distance=cosine_nclusters=100.pkl --style-ext .png --num-clusters 100

import os, sys
from argparse import ArgumentParser
import re
import pickle
import numpy as np
import random
import json


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--file-pairs-id-subj',  type=str, default='')   # /datasets2/bjgbiesseck/face_recognition/synthetic/dcface_with_pretrained_models/dcface_original_synthetic_ids/dcface_original_10000_synthetic_ids_STYLE_FEATURES_CLUSTERING_FROM_1_CASIA-WebFace-imgs_crops_112x112_STYLE_FEATURES_CLUSTERING-feature=_style-_distance=cosine-nclusters=100/feature=_style/_distance=cosine/nclusters=100/face_pairs_subj_style_by_race.json
    parser.add_argument('--dataset-path',        type=str, default='')   # /datasets2/bjgbiesseck/face_recognition/dcface/generated_images/tcdiff_WITH_BFM_e:10_spatial_dim:5_bias:0.0_casia_ir50_09-10_1_EQUALIZED-STYLES_BY-RACE_NCLUSTERS=100
    parser.add_argument('--ext',        type=str, default='.jpg')

    args = parser.parse_args()
    return args


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_all_files_in_path(folder_path, file_extension=['_spatial.pt'], pattern=''):
    if not type(file_extension) is list: file_extension = [file_extension]
    file_list = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            path_file = os.path.join(root, filename)
            for ext in file_extension:
                if pattern in path_file and path_file.lower().endswith(ext.lower()):
                    file_list.append(path_file)
                    # print(f'{len(file_list)}', end='\r')
    print()
    # file_list.sort()
    file_list = natural_sort(file_list)
    return file_list


def get_subfolders(directory):
    subfolders = [os.path.join(directory, f.name) for f in os.scandir(directory) if f.is_dir()]
    subfolders = natural_sort(subfolders)
    return subfolders


def save_dict(data: dict, path: str) -> None:
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load_dict(path: str) -> dict:
    with open(path, 'rb') as file:
        return pickle.load(file)


def save_dict_to_json(data: dict, filename: str, indent: int = 4):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def make_dict_of_clusters_labels(clusters_data):
    cluster_ids        = clusters_data['cluster_ids']
    corresp_imgs_paths = clusters_data['corresp_imgs_paths']

    cluster_ids_dict_idx = {}
    cluster_ids_dict_paths = {}
    for idx, label in enumerate(cluster_ids):
        label = int(label)
        # print('idx:', idx, '    label:', int(label))
        if not label in cluster_ids_dict_idx:
            cluster_ids_dict_idx[label]   = []
            cluster_ids_dict_paths[label] = []
        
        cluster_ids_dict_idx[label].append(idx)
        cluster_ids_dict_paths[label].append(corresp_imgs_paths[idx])
    
    return cluster_ids_dict_idx, cluster_ids_dict_paths


def make_dict_of_clusters_labels_by_ethnic_groups(clusters_data):
    cluster_ids        = clusters_data['cluster_ids']
    corresp_imgs_paths = clusters_data['corresp_imgs_paths']
    dominant_races     = clusters_data['dominant_races']

    cluster_ids_races_dict_idx = {}
    cluster_ids_races_dict_paths = {}
    for idx, (race, label) in enumerate(zip(dominant_races, cluster_ids)):
        if not race in cluster_ids_races_dict_idx:
            cluster_ids_races_dict_idx[race]   = {}
            cluster_ids_races_dict_paths[race] = {}

        label = int(label)
        if not label in cluster_ids_races_dict_idx[race]:
            cluster_ids_races_dict_idx[race][label]   = []
            cluster_ids_races_dict_paths[race][label] = []
        
        cluster_ids_races_dict_idx[race][label].append(idx)
        cluster_ids_races_dict_paths[race][label].append(corresp_imgs_paths[idx])

    return cluster_ids_races_dict_idx, cluster_ids_races_dict_paths


def find_lowest_indices(lst, M):
    # return np.argsort(lst)[:M].tolist()
    return np.argsort(lst)[:M]


def select_style_samples_from_lowest_clusters(subj_indices_lowest_styles, style_dict_paths_cluster_ids, num_samples_per_cluster=1):
    all_selected_style_samples = {}
    for idx_cluster, cluster in enumerate(subj_indices_lowest_styles):
        # print(f"{idx_cluster} - cluster: {cluster} - {style_dict_paths_cluster_ids[cluster]}")
        # sys.exit(0)
        selected_style_samples_one_cluster = random.sample(style_dict_paths_cluster_ids[cluster], num_samples_per_cluster)
        assert not cluster in all_selected_style_samples, f"Error, cluster key '{cluster}' already in 'all_selected_style_samples'" 
        all_selected_style_samples[cluster] = selected_style_samples_one_cluster
    
    assert len(all_selected_style_samples) == len(subj_indices_lowest_styles), \
        f"Error, len(all_selected_style_samples) ({len(all_selected_style_samples)}) != len(subj_indices_lowest_styles) ({len(subj_indices_lowest_styles)})"
    return all_selected_style_samples


def select_style_samples_from_lowest_clusters_by_race(subj_indices_lowest_styles, subj_dominant_race, style_dict_paths_cluster_ids, num_samples_per_cluster=1):
    all_selected_style_samples = {}
    for idx_cluster, cluster in enumerate(subj_indices_lowest_styles):
        # print(f"{idx_cluster} - cluster: {cluster} - {style_dict_paths_cluster_ids[cluster]}")
        # sys.exit(0)
        selected_style_samples_one_cluster = random.sample(style_dict_paths_cluster_ids[subj_dominant_race][cluster], num_samples_per_cluster)
        assert not cluster in all_selected_style_samples, f"Error, cluster key '{cluster}' already in 'all_selected_style_samples'" 
        all_selected_style_samples[cluster] = selected_style_samples_one_cluster
    
    assert len(all_selected_style_samples) == len(subj_indices_lowest_styles), \
        f"Error, len(all_selected_style_samples) ({len(all_selected_style_samples)}) != len(subj_indices_lowest_styles) ({len(subj_indices_lowest_styles)})"
    return all_selected_style_samples




def main(args):
    if args.file_pairs_id_subj:
        print(f'Loading \'{args.file_pairs_id_subj}\'')
        all_face_pairs_subj_style = load_json(args.file_pairs_id_subj)
        # print('all_face_pairs_subj_style:', all_face_pairs_subj_style)

        for idx_subj, subj_path in enumerate(list(all_face_pairs_subj_style.keys())):
            print('subj_path:', subj_path)
            
            subj_facial_attrib_parent_dir = '/'.join(subj_path.split('/')[:-2]) + '_FACE_ATTRIB'
            assert os.path.isdir(subj_facial_attrib_parent_dir)
            # print('subj_facial_attrib_parent_dir:', subj_facial_attrib_parent_dir)

            subj_facial_attrib_subpath = '/'.join(subj_path.split('/')[-2:]).replace('.png', '.pkl')
            # print('subj_facial_attrib_subpath:', subj_facial_attrib_subpath)

            subj_facial_attrib_full_path = os.path.join(subj_facial_attrib_parent_dir, subj_facial_attrib_subpath)
            assert os.path.isfile(subj_facial_attrib_full_path)
            print('subj_facial_attrib_full_path:', subj_facial_attrib_full_path)

            subj_face_attribs = load_dict(subj_facial_attrib_full_path)
            # print('subj_face_attribs:', subj_face_attribs)
            subj_dominant_race = subj_face_attribs['race']['dominant_race']
            print('subj_dominant_race:', subj_dominant_race)

            style_paths = all_face_pairs_subj_style[subj_path]
            style_dominant_races = []
            for idx_sty, sty_path in enumerate(style_paths):
                # print('sty_path:', sty_path)
                sty_facial_attrib_parent_dir = '/'.join(sty_path.split('/')[:-2]) + '_FACE_ATTRIB'
                assert os.path.isdir(sty_facial_attrib_parent_dir)
                sty_facial_attrib_subpath = '/'.join(sty_path.split('/')[-2:]).replace('.png', '.pkl')
                sty_facial_attrib_full_path = os.path.join(sty_facial_attrib_parent_dir, sty_facial_attrib_subpath)
                assert os.path.isfile(sty_facial_attrib_full_path)
                sty_face_attribs = load_dict(sty_facial_attrib_full_path)
                sty_dominant_race = sty_face_attribs['race']['dominant_race']
                style_dominant_races.append(sty_dominant_race)
            print('style_dominant_races:', style_dominant_races)

            print('-------')

            input('PAUSE')
            # sys.exit(0)


    elif args.dataset_path:
        print(f'Searching subfolders in \'{args.dataset_path}\'')
        subfolders = get_subfolders(args.dataset_path)
        print(f'Found {len(subfolders)} subfolders\n------------------')

        for idx_subfolder, path_subfolder in enumerate(subfolders):
            files_subfolder = get_all_files_in_path(path_subfolder, args.ext)
            # print('files_subfolder:', files_subfolder)
            print('path_subfolder:', path_subfolder)

            style_dominant_races = []
            for idx_file, file_path in enumerate(files_subfolder):
                sty_facial_attrib_parent_dir = '/'.join(file_path.split('/')[:-2]) + '_FACE_ATTRIB'
                assert os.path.isdir(sty_facial_attrib_parent_dir)
                sty_facial_attrib_subpath = '/'.join(file_path.split('/')[-2:]).replace('.jpg', '.pkl')
                sty_facial_attrib_full_path = os.path.join(sty_facial_attrib_parent_dir, sty_facial_attrib_subpath)
                assert os.path.isfile(sty_facial_attrib_full_path)
                sty_face_attribs = load_dict(sty_facial_attrib_full_path)
                sty_dominant_race = sty_face_attribs['race']['dominant_race']
                # print('sty_facial_attrib_full_path:', sty_facial_attrib_full_path, '    dominant_race:', sty_dominant_race)
                style_dominant_races.append(sty_dominant_race)
                # print(f"file_path: {file_path} - {sty_dominant_race}")
            print('style_dominant_races:', style_dominant_races)

            print('-------')
            input('PAUSE')

            # sys.exit(0)



if __name__ == "__main__":
    args = parse_args()
    main(args)