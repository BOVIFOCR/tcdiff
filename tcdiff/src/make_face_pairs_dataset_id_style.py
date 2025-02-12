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
    parser.add_argument('--subj-clusters',         type=str, default='/datasets2/bjgbiesseck/face_recognition/synthetic/dcface_with_pretrained_models/dcface_original_synthetic_ids/dcface_original_10000_synthetic_ids_STYLE_FEATURES_CLUSTERING_FROM_1_CASIA-WebFace-imgs_crops_112x112_STYLE_FEATURES_CLUSTERING-feature=_style-_distance=cosine-nclusters=100/feature=_style/_distance=cosine/nclusters=100/clusters-data_feature=_style.pt_distance=cosine_nclusters=100.pkl', help='')
    # parser.add_argument('--id-clusters-imgs',    type=str, default='/datasets2/bjgbiesseck/face_recognition/synthetic/dcface_with_pretrained_models/dcface_original_synthetic_ids/dcface_original_10000_synthetic_ids_STYLE_FEATURES_CLUSTERING_FROM_1_CASIA-WebFace-imgs_crops_112x112_STYLE_FEATURES_CLUSTERING-feature=_style-_distance=cosine-nclusters=100/feature=_style/_distance=cosine/nclusters=100/clusters_imgs')
    parser.add_argument('--subj-ext',              type=str, default='.png')
    parser.add_argument('--num-samples-per-id',  type=int, default=49)
    parser.add_argument('--style-clusters',      type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112_STYLE_FEATURES_CLUSTERING/feature=_style/_distance=cosine/nclusters=100/clusters-data_feature=_style.pt_distance=cosine_nclusters=100.pkl', help='')
    # parser.add_argument('--style-clusters-imgs', type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112_STYLE_FEATURES_CLUSTERING/feature=_style/_distance=cosine/nclusters=100/clusters_imgs')
    parser.add_argument('--style-ext',           type=str, default='.png')
    parser.add_argument('--num-clusters',        type=int, default=100)
    parser.add_argument('--output-dir',          type=str, default='')

    # parser.add_argument('--input', type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_crops_112x112_JUST-PROTOCOL-IMGS_STYLE_FEATURES')
    # parser.add_argument('--ext', type=str, default='_style.pt')
    # parser.add_argument('--corresponding-imgs', type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_crops_112x112_JUST-PROTOCOL-IMGS', help='')
    # parser.add_argument('--num-imgs-clusters-to-save', type=int, default=0)
    # parser.add_argument('--num-clusters', type=int, default=100)
    # parser.add_argument('--distance', type=str, default='cosine', help='cosine or euclidean')
    # parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda:0')
    # parser.add_argument('--facial-attributes', type=str, default='', help='')    # '/datasets2/1st_frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_crops_112x112_JUST-PROTOCOL-IMGS_FACE_ATTRIB'
    # parser.add_argument('--source-clusters', type=str, default='', help='')      # '/datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112_STYLE_FEATURES_CLUSTERING/feature=_style/_distance=cosine/nclusters=100/clusters-data_feature=_style.pt_distance=cosine_nclusters=100.pkl'

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
                    print(f'{len(file_list)}', end='\r')
    print()
    # file_list.sort()
    file_list = natural_sort(file_list)
    return file_list


def save_dict(data: dict, path: str) -> None:
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load_dict(path: str) -> dict:
    with open(path, 'rb') as file:
        return pickle.load(file)


def save_dict_to_json(data: dict, filename: str, indent: int = 4):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


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
    print('--------------')
    print(f'Loading subj-clusters: \'{args.subj_clusters}\'')
    subj_clusters_data = load_dict(args.subj_clusters)
    print('Loaded subj_clusters_data.keys():', subj_clusters_data.keys())

    for idx_race, race in enumerate(list(subj_clusters_data['races_styles_clusters_count'].keys())):
        subj_clusters_data['races_styles_clusters_count'][race] = np.array(subj_clusters_data['races_styles_clusters_count'][race])

    print('\nMaking dict of subj clusters labels...')
    # subj_dict_idx_cluster_ids, subj_dict_paths_cluster_ids = make_dict_of_clusters_labels(subj_clusters_data)
    subj_dict_idx_cluster_ids, subj_dict_paths_cluster_ids = make_dict_of_clusters_labels_by_ethnic_groups(subj_clusters_data)
    subj_clusters_data['dict_idx_cluster_ids']   = subj_dict_idx_cluster_ids
    subj_clusters_data['dict_paths_cluster_ids'] = subj_dict_paths_cluster_ids
    # print("id_clusters_data['cluster_ids_dict_paths']:", id_clusters_data['cluster_ids_dict_paths'])
    print('Done')


    print('--------------')
    print(f'Loading style-clusters: \'{args.style_clusters}\'')
    style_clusters_data = load_dict(args.style_clusters)
    print('Loaded style_clusters_data.keys():', style_clusters_data.keys())

    for idx_race, race in enumerate(list(style_clusters_data['races_styles_clusters_count'].keys())):
        style_clusters_data['races_styles_clusters_count'][race] = np.array(style_clusters_data['races_styles_clusters_count'][race])

    print('\nMaking dict of style clusters labels...')
    # style_dict_idx_cluster_ids, style_dict_paths_cluster_ids = make_dict_of_clusters_labels(style_clusters_data)
    style_dict_idx_cluster_ids, style_dict_paths_cluster_ids = make_dict_of_clusters_labels_by_ethnic_groups(style_clusters_data)
    style_clusters_data['dict_idx_cluster_ids']   = style_dict_idx_cluster_ids
    style_clusters_data['dict_paths_cluster_ids'] = style_dict_paths_cluster_ids
    print('Done')



    print('--------------')
    all_face_pairs_subj_style = {}
    total_subj_and_style_samples = 0
    for idx_subj_dominant_race, (subj_img_path, subj_dominant_race) in enumerate(zip(subj_clusters_data['corresp_imgs_paths'], subj_clusters_data['dominant_races'])):
        print(f"{idx_subj_dominant_race}/{len(subj_clusters_data['dominant_races'])} - subj_dominant_race: {subj_dominant_race}")
        print(f"subj_img_path: {subj_img_path}")
        subj_race_styles_clusters_count = subj_clusters_data['races_styles_clusters_count'][subj_dominant_race]
        subj_indices_lowest_styles = find_lowest_indices(subj_race_styles_clusters_count, args.num_samples_per_id)
        # print('subj_indices_lowest_styles:', subj_indices_lowest_styles, '    type:', type(subj_indices_lowest_styles))

        # style_selected_samples_paths = select_style_samples_from_lowest_clusters(subj_indices_lowest_styles, style_dict_paths_cluster_ids)
        style_selected_samples_paths = select_style_samples_from_lowest_clusters_by_race(subj_indices_lowest_styles, subj_dominant_race, style_dict_paths_cluster_ids)
        # print('style_selected_samples_paths:', style_selected_samples_paths)

        print(f"Selected style images: {len(list(style_selected_samples_paths.keys()))}")
        list_style_selected_samples_paths = []
        for idx_samples_paths, samples_paths in enumerate(list(style_selected_samples_paths.values())):
            list_style_selected_samples_paths.extend(samples_paths)
        # print('list_style_selected_samples_paths:', list_style_selected_samples_paths)
        # sys.exit(0)

        all_face_pairs_subj_style[subj_img_path] = list_style_selected_samples_paths
        subj_race_styles_clusters_count[subj_indices_lowest_styles] += len(style_selected_samples_paths[list(style_selected_samples_paths.keys())[0]])

        total_subj_and_style_samples += len(list_style_selected_samples_paths) + 1   # Adds 1 due the original subject sample
        print('total_subj_and_style_samples:', total_subj_and_style_samples)
        print('-----')

    # print("subj_clusters_data['races_styles_clusters_count']:", subj_clusters_data['races_styles_clusters_count'])

    # face_pairs_file_path = os.path.join(os.path.dirname(args.subj_clusters), 'face_pairs_subj_style.json')
    face_pairs_file_path = os.path.join(os.path.dirname(args.subj_clusters), 'face_pairs_subj_style_by_race.json')
    print(f"Saving face pairs to disk: {face_pairs_file_path}")
    save_dict_to_json(all_face_pairs_subj_style, face_pairs_file_path, indent=4)

    print('\nFinished!')



if __name__ == "__main__":
    args = parse_args()
    main(args)