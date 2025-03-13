# duo
# export CUDA_VISIBLE_DEVICES=0; python select_generated_samples_oversampled_create_dataset.py --file-pairs-id-subj /datasets2/bjgbiesseck/face_recognition/synthetic/dcface_with_pretrained_models/dcface_original_synthetic_ids/dcface_original_10000_synthetic_ids_STYLE_FEATURES_CLUSTERING_FROM_1_CASIA-WebFace-imgs_crops_112x112_STYLE_FEATURES_CLUSTERING-feature=_style-_distance=cosine-nclusters=100/feature=_style/_distance=cosine/nclusters=100/face-pairs-subj-style-by-race_num-samples-per-id=49_num-samples-per-cluster=20.json --corresponding-subj-clusters /datasets2/bjgbiesseck/face_recognition/synthetic/dcface_with_pretrained_models/dcface_original_synthetic_ids/dcface_original_10000_synthetic_ids_STYLE_FEATURES_CLUSTERING_FROM_1_CASIA-WebFace-imgs_crops_112x112_STYLE_FEATURES_CLUSTERING-feature=_style-_distance=cosine-nclusters=100/feature=_style/_distance=cosine/nclusters=100/clusters-data_feature=_style.pt_distance=cosine_nclusters=100.pkl --corresponding-style-clusters /datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112_STYLE_FEATURES_CLUSTERING/feature=_style/_distance=cosine/nclusters=100/clusters-data_feature=_style.pt_distance=cosine_nclusters=100.pkl --stylized-imgs /datasets3/bjgbiesseck/face_recognition/dcface/generated_images/tcdiff_WITH_BFM_e:10_spatial_dim:5_bias:0.0_casia_ir50_09-10_1_EQUALIZED-STYLES_BY-RACE_NCLUSTERS=100_ALLOW-STYLE-REPEAT_part=0-4 --corresp-style-features /datasets3/bjgbiesseck/face_recognition/dcface/generated_images/tcdiff_WITH_BFM_e:10_spatial_dim:5_bias:0.0_casia_ir50_09-10_1_EQUALIZED-STYLES_BY-RACE_NCLUSTERS=100_ALLOW-STYLE-REPEAT_part=0-4_STYLE_FEATURES --num-samples-per-subj 49 --num-samples-per-cluster 20


import os, sys
from argparse import ArgumentParser
import re
import pickle
import numpy as np
import random
import json
import torch
from scipy.stats import entropy
import matplotlib.pyplot as plt
import time
import kmeans_pytorch


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--file-pairs-id-subj', type=str, default='')                     # /datasets2/bjgbiesseck/face_recognition/synthetic/dcface_with_pretrained_models/dcface_original_synthetic_ids/dcface_original_10000_synthetic_ids_STYLE_FEATURES_CLUSTERING_FROM_1_CASIA-WebFace-imgs_crops_112x112_STYLE_FEATURES_CLUSTERING-feature=_style-_distance=cosine-nclusters=100/feature=_style/_distance=cosine/nclusters=100/face-pairs-subj-style-by-race_num-samples-per-id=49_num-samples-per-cluster=20.json
    parser.add_argument('--corresponding-subj-clusters', type=str, default='', help='')   # /datasets2/bjgbiesseck/face_recognition/synthetic/dcface_with_pretrained_models/dcface_original_synthetic_ids/dcface_original_10000_synthetic_ids_STYLE_FEATURES_CLUSTERING_FROM_1_CASIA-WebFace-imgs_crops_112x112_STYLE_FEATURES_CLUSTERING-feature=_style-_distance=cosine-nclusters=100/feature=_style/_distance=cosine/nclusters=100/clusters-data_feature=_style.pt_distance=cosine_nclusters=100.pkl
    parser.add_argument('--corresponding-style-clusters', type=str, default='', help='')  # /datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112_STYLE_FEATURES_CLUSTERING/feature=_style/_distance=cosine/nclusters=100/clusters-data_feature=_style.pt_distance=cosine_nclusters=100.pkl
    parser.add_argument('--stylized-imgs', type=str, default='', help='')                 # /datasets3/bjgbiesseck/face_recognition/dcface/generated_images/tcdiff_WITH_BFM_e:10_spatial_dim:5_bias:0.0_casia_ir50_09-10_1_EQUALIZED-STYLES_BY-RACE_NCLUSTERS=100_ALLOW-STYLE-REPEAT_part=0-4
    parser.add_argument('--corresp-style-features', type=str, default='', help='')        # /datasets3/bjgbiesseck/face_recognition/dcface/generated_images/tcdiff_WITH_BFM_e:10_spatial_dim:5_bias:0.0_casia_ir50_09-10_1_EQUALIZED-STYLES_BY-RACE_NCLUSTERS=100_ALLOW-STYLE-REPEAT_part=0-4_STYLE_FEATURES

    parser.add_argument('--num-samples-per-subj', type=int, default=49, help='')
    parser.add_argument('--num-samples-per-cluster', type=int, default=20, help='')

    parser.add_argument('--start-subj', type=str, default='')   # example '0'

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


def compute_statistical_metrics(normalized_data):
    values_sum = np.sum(normalized_data)
    min_allowed, max_allowed = 0.99, 1.001
    assert values_sum == 0.0 or (values_sum >= min_allowed and values_sum <= max_allowed), f'np.sum(normalized_data) is {values_sum}, should be in [{max_allowed}, {min_allowed}]'
    stats = {}

    mean, std_dev = np.mean(normalized_data), np.std(normalized_data)
    # stats['mean'] = mean
    # stats['std'] = std_dev

    normalized_entropy = 0.0
    if values_sum > 0.0:
        entropy_value = entropy(normalized_data, base=2)
        normalized_entropy = entropy_value / np.log2(len(normalized_data))
    stats['entropy'] = normalized_entropy

    # gini_index = 1 - np.sum(normalized_data**2)
    # stats['gini'] = gini_index

    cv = 0.0
    if std_dev > 0 and mean > 0:
        cv = std_dev / mean
    stats['cv'] = cv

    uniform_prob = np.ones_like(normalized_data) / len(normalized_data)
    normalized_data = normalized_data[normalized_data > 0]
    uniform_prob = uniform_prob[:len(normalized_data)]
    kl_div = np.sum(normalized_data * (np.log2(normalized_data) - np.log2(uniform_prob)))
    stats['kl_div'] = kl_div

    return stats


def save_styles_per_race_bars_chart(ndarrays, ndarrays_stats, global_title, output_path):
    races = list(ndarrays.keys())
    ndarrays = [ndarrays[race] for race in races]
    stats = [ndarrays_stats[race] for race in races]

    if len(ndarrays) != len(races) or len(stats) != len(races):
        raise ValueError("The number of ndarrays and stats must match the number of subtitles.")

    # Set the global maximum value for consistent y-axis scaling
    # global_max = 0.05  # 5%
    # global_max = 0.1   # 10%
    global_max = 1/len(ndarrays[0]) * 5

    n_subplots = len(ndarrays)
    fig_height = 12
    fig, axes = plt.subplots(n_subplots, 2, figsize=(16, fig_height), constrained_layout=True, 
                              gridspec_kw={"width_ratios": [3, 1]})

    if n_subplots == 1:
        axes = [axes]

    fig.suptitle(global_title, fontsize=16, weight='bold')

    for i, ((bar_ax, stat_ax), arr, stat, subtitle) in enumerate(zip(axes, ndarrays, stats, races)):
        # Plot bar chart for ndarrays
        bar_ax.bar(range(len(arr)), arr)
        bar_ax.set_ylim(0, global_max)
        bar_ax.set_yticks([0, global_max])
        bar_ax.set_title(f'{subtitle} (styles)', fontsize=14)
        if i == len(ndarrays) - 1:
            bar_ax.set_xlabel("Face Styles", fontsize=12)
        bar_ax.set_ylabel("Percentual", fontsize=12)

        # Set x-ticks and labels for bar_ax
        bar_ax.set_xticks(range(len(arr)))
        bar_ax.set_xticklabels(range(len(arr)), fontsize=8, rotation=90)

        # Plot vertical bar chart for statistics
        stat_labels = list(stat.keys())
        stat_values = list(stat.values())
        bars = stat_ax.bar(stat_labels, stat_values, color="orange")
        stat_ax.set_title(f'{subtitle} (statistics)', fontsize=14)
        stat_ax.set_ylim(0, 2)
        stat_ax.set_ylabel("Value", fontsize=10)

        # Add value annotations to the bars
        for bar, value in zip(bars, stat_values):
            stat_ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f'{value:.3f}', 
                         ha='center', va='bottom', fontsize=14)

    plt.savefig(output_path, format='png')
    plt.close(fig)




def main(args):

    print('-----------')
    print(f'Loading \'{args.file_pairs_id_subj}\'')
    all_face_pairs_subj_style = load_json(args.file_pairs_id_subj)
    # print('all_face_pairs_subj_style:', all_face_pairs_subj_style)

    # subj_clusters_data.keys(): dict_keys(['files_paths', 'original_feats', 'cluster_ids', 'feats_tsne', 'cluster_centers_tsne', 'corresp_imgs_paths', 'facial_attribs_paths', 'facial_attribs', 'dominant_races', 'races_styles_clusters_count'])
    print(f'\nLoading subj-clusters: \'{args.corresponding_subj_clusters}\'')
    subj_clusters_data = load_dict(args.corresponding_subj_clusters)
    print('Loaded subj_clusters_data.keys():', subj_clusters_data.keys())
    subj_corresp_imgs_paths = subj_clusters_data['corresp_imgs_paths']
    subj_cluster_ids = subj_clusters_data['cluster_ids']
    # print('subj_corresp_imgs_paths:', subj_corresp_imgs_paths)
    # sys.exit(0)

    # style_clusters_data.keys(): dict_keys(['files_paths', 'original_feats', 'cluster_ids', 'cluster_centers', 'feats_tsne', 'cluster_centers_tsne', 'corresp_imgs_paths', 'facial_attribs_paths', 'facial_attribs', 'dominant_races', 'races_styles_clusters_count'])
    print(f'\nLoading style-clusters: \'{args.corresponding_style_clusters}\'')
    style_clusters_data = load_dict(args.corresponding_style_clusters)
    print('Loaded style_clusters_data.keys():', style_clusters_data.keys())
    print()
    style_corresp_imgs_paths = style_clusters_data['corresp_imgs_paths']
    style_cluster_ids = style_clusters_data['cluster_ids']
    cluster_centers = style_clusters_data['cluster_centers']
    cluster_centers = torch.from_numpy(cluster_centers)

    # races_list = list(set(subj_clusters_data['dominant_races']))
    races_list = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']
    # print('races_list:', races_list)
    races_styles_clusters_count = {race: np.zeros((len(subj_clusters_data['cluster_centers_tsne']),)) for race in races_list}

    starting_index_subj = 0
    if args.start_subj:
        for i, subj_path in enumerate(all_face_pairs_subj_style.keys()):
            if subj_path.split('/')[-2] == args.start_subj:
                starting_index_subj = i

    num_samples_per_selected_cluster = args.num_samples_per_cluster
    total_num_styles_at_least_one_hit = 0
    total_num_subj_having_all_styles_hit = 0
    total_elapsed_time = 0.0

    for idx_orig_subj_path, orig_subj_path in enumerate(all_face_pairs_subj_style.keys()):
        if idx_orig_subj_path >= starting_index_subj:
            start_time = time.time()

            print(f'{idx_orig_subj_path} orig_subj_path: {orig_subj_path}')
            orig_styles_paths = all_face_pairs_subj_style[orig_subj_path]
            
            # subj_all_orig_style_ids = np.zeros(args.num_samples_per_subj*num_samples_per_selected_cluster, dtype=int)
            subj_all_orig_style_ids = torch.zeros(args.num_samples_per_subj*num_samples_per_selected_cluster, dtype=int, device=torch.device('cuda:0'))
            for idx_orig_style_path, orig_style_path in enumerate(orig_styles_paths):
                index_style_sample = style_clusters_data['corresp_imgs_paths'].index(orig_style_path)
                orig_style_cluster = style_clusters_data['cluster_ids'][index_style_sample]
                subj_all_orig_style_ids[idx_orig_style_path] = orig_style_cluster
                # print('orig_style_path:', orig_style_path, '    orig_style_cluster:', orig_style_cluster)
            
            subj_name = orig_subj_path.split('/')[-2]
            stylized_sample_index = 0
            subj_samples_feats = torch.zeros((args.num_samples_per_subj*num_samples_per_selected_cluster,128), dtype=torch.float, device=torch.device('cuda:0'))
            for idx_orig_style_path, orig_style_path in enumerate(orig_styles_paths):
                generated_stylized_img_name = f'{stylized_sample_index}.jpg'
                generated_stylized_feature_name = f'{stylized_sample_index}_style.pt'
                generated_stylized_feature_path = os.path.join(args.corresp_style_features, subj_name, generated_stylized_feature_name)
                generated_stylized_feature = torch.load(generated_stylized_feature_path)
                generated_stylized_feature = torch.flatten(generated_stylized_feature, start_dim=1)
                subj_samples_feats[idx_orig_style_path,:] = generated_stylized_feature
                # print(f'idx_orig_style_path: {idx_orig_style_path}', '    generated_stylized_feature.shape:', generated_stylized_feature.shape)
                stylized_sample_index += 1
                # sys.exit(0)
            
            # print('Predicting samples ids from clusters...')
            generated_stylized_ids = kmeans_pytorch.kmeans_predict(X=subj_samples_feats,
                                                            cluster_centers=cluster_centers,
                                                            distance='cosine',
                                                            device=torch.device('cuda:0'),
                                                            gamma_for_soft_dtw=0.001,
                                                            tqdm_flag=True).to(torch.device('cuda:0'))
            # print('generated_stylized_ids:', generated_stylized_ids)
            # print('type(generated_stylized_ids):', type(generated_stylized_ids))
            clusters_ids_comparison = torch.eq(subj_all_orig_style_ids, generated_stylized_ids)
            subj_num_hits = torch.sum(clusters_ids_comparison)
            subj_num_misses = len(clusters_ids_comparison) - subj_num_hits
            print(f'subj_num_hits: {subj_num_hits}    subj_num_misses: {subj_num_misses}    total_samples_per_subj: {subj_num_hits+subj_num_misses}')

            num_styles_at_least_one_hit = 0
            for idx_part in range(args.num_samples_per_subj):
                idx_begin = idx_part * num_samples_per_selected_cluster
                idx_end   = idx_begin + num_samples_per_selected_cluster
                # print('subj_all_orig_style_ids[idx_begin:idx_end]:', subj_all_orig_style_ids[idx_begin:idx_end])
                # print('generated_stylized_ids[idx_begin:idx_end]:', generated_stylized_ids[idx_begin:idx_end])
                style_ids_comparison = torch.eq(subj_all_orig_style_ids[idx_begin:idx_end], generated_stylized_ids[idx_begin:idx_end])
                subj_num_hits = torch.sum(style_ids_comparison)
                if subj_num_hits > 0:
                    num_styles_at_least_one_hit += 1
            if num_styles_at_least_one_hit == args.num_samples_per_subj: total_num_subj_having_all_styles_hit += 1
            total_num_styles_at_least_one_hit += num_styles_at_least_one_hit
            total_mean_num_styles_at_least_one_hit = total_num_styles_at_least_one_hit/((idx_orig_subj_path-starting_index_subj)+1)
            print(f'num_styles_at_least_one_hit:', num_styles_at_least_one_hit)
            print(f'    total_mean_num_styles_at_least_one_hit: {total_mean_num_styles_at_least_one_hit}/{args.num_samples_per_subj} = {total_mean_num_styles_at_least_one_hit/float(args.num_samples_per_subj)}')
            print(f'    total_num_subj_having_all_styles_hit: {total_num_subj_having_all_styles_hit}/{((idx_orig_subj_path-starting_index_subj)+1)} = {total_num_subj_having_all_styles_hit/((idx_orig_subj_path-starting_index_subj)+1)}')

            elapsed_time = time.time()-start_time
            total_elapsed_time += elapsed_time
            avg_sample_time = total_elapsed_time / ((idx_orig_subj_path-starting_index_subj)+1)
            estimated_time = avg_sample_time * (len(all_face_pairs_subj_style.keys())-((idx_orig_subj_path-starting_index_subj)+1))
            print("    Elapsed time: %.3fs" % elapsed_time)
            print("    Avg elapsed time: %.3fs" % avg_sample_time)
            print("    Total elapsed time: %.3fs,  %.3fm,  %.3fh" % (total_elapsed_time, total_elapsed_time/60, total_elapsed_time/3600))
            print("    Estimated Time to Completion (ETC): %.3fs,  %.3fm,  %.3fh" % (estimated_time, estimated_time/60, estimated_time/3600))
            print('--------------')
            # sys.exit(0)

        else:
            print(f"Skiping idx: {idx_orig_subj_path} - subj: {orig_subj_path}", end='\r')

    print('\nFinished!')


if __name__ == "__main__":
    args = parse_args()
    main(args)