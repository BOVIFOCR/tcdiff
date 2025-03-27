## duo
# duo
# export CUDA_VISIBLE_DEVICES=1; python cluster_classify_style_faces.py --input /datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112_STYLE_FEATURES --ext _style.pt --corresponding-imgs /datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112 --num-imgs-clusters-to-save 10 --num-clusters 100 --distance cosine --device cpu --facial-attributes /datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112_FACE_ATTRIB
# export CUDA_VISIBLE_DEVICES=1; python cluster_classify_style_faces.py --input /datasets2/1st_frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_crops_112x112_JUST-PROTOCOL-IMGS_STYLE_FEATURES --ext _style.pt --corresponding-imgs /datasets2/1st_frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_crops_112x112_JUST-PROTOCOL-IMGS --num-imgs-clusters-to-save 10 --num-clusters 200 --distance cosine --device cpu --facial-attributes /datasets2/1st_frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_crops_112x112_JUST-PROTOCOL-IMGS_FACE_ATTRIB --source-clusters /datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112_STYLE_FEATURES_CLUSTERING/feature=_style/_distance=cosine/nclusters=200/clusters-data_feature=_style.pt_distance=cosine_nclusters=200.pkl
# export CUDA_VISIBLE_DEVICES=1; python cluster_classify_style_faces.py --input /datasets2/bjgbiesseck/face_recognition/dcface/generated_images/tcdiff_WITH_BFM_e:10_spatial_dim:5_bias:0.0_casia_ir50_09-10_1_EQUALIZED-STYLES_NCLUSTERS=100_STYLE_FEATURES --ext _style.pt --corresponding-imgs /datasets2/bjgbiesseck/face_recognition/dcface/generated_images/tcdiff_WITH_BFM_e:10_spatial_dim:5_bias:0.0_casia_ir50_09-10_1_EQUALIZED-STYLES_NCLUSTERS=100 --num-imgs-clusters-to-save 10 --num-clusters 100 --distance cosine --device cpu --facial-attributes /datasets2/bjgbiesseck/face_recognition/dcface/generated_images/tcdiff_WITH_BFM_e:10_spatial_dim:5_bias:0.0_casia_ir50_09-10_1_EQUALIZED-STYLES_NCLUSTERS=100_FACE_ATTRIB --source-clusters /datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112_STYLE_FEATURES_CLUSTERING/feature=_style/_distance=cosine/nclusters=100/clusters-data_feature=_style.pt_distance=cosine_nclusters=100.pkl

import os, sys
from argparse import ArgumentParser
import re

import torch
import numpy as np
import glob
import shutil
import pickle
import kmeans_pytorch
import tsnecuda
# from tsnecuda import TSNE
from scipy.stats import entropy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# from collections import Counter


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_crops_112x112_JUST-PROTOCOL-IMGS_STYLE_FEATURES')
    parser.add_argument('--ext', type=str, default='_style.pt')
    parser.add_argument('--corresponding-imgs', type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_crops_112x112_JUST-PROTOCOL-IMGS', help='')
    parser.add_argument('--num-imgs-clusters-to-save', type=int, default=0)
    parser.add_argument('--num-clusters', type=int, default=100)
    parser.add_argument('--distance', type=str, default='cosine', help='cosine or euclidean')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda:0')
    parser.add_argument('--facial-attributes', type=str, default='', help='')    # '/datasets2/1st_frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_crops_112x112_JUST-PROTOCOL-IMGS_FACE_ATTRIB'
    parser.add_argument('--source-clusters', type=str, default='', help='')      # '/datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112_STYLE_FEATURES_CLUSTERING/feature=_style/_distance=cosine/nclusters=100/clusters-data_feature=_style.pt_distance=cosine_nclusters=100.pkl'

    parser.add_argument('--uniform-selection', type=float, default=1.0)

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


def save_scatter_plot_clusters(data, labels, centers, title='', output_path=''):
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=30000/len(data), cmap="viridis")
    plt.scatter(centers[:, 0], centers[:, 1], c="black", s=50, alpha=0.8)

    for i in range(len(centers)):
        plt.annotate(str(i), (centers[i,0], centers[i,1]))

    plt.tight_layout()
    plt.savefig(output_path)
    # plt.close(fig)
    # sys.exit(0)


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


def save_races_count_bar_chart(race_list, races_labels_dict, title, output_path):
    races_idx = np.array([races_labels_dict[race] for race in race_list])
    counts = [sum(races_idx == race_idx) for race_idx in list(races_labels_dict.values())]
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(list(races_labels_dict.keys()), counts)
    ax.set_xlabel('Race')
    ax.set_ylabel('Number of Samples')
    ax.set_title(title)

    ax.set_xticks(range(len(races_labels_dict)))
    ax.set_xticklabels(list(races_labels_dict.keys()), rotation=45, ha='right')
    
    plt.tight_layout()
    
    plt.savefig(output_path, format='png')
    plt.close(fig)




def main(args):

    if not args.source_clusters:
        output_dir = args.input.rstrip('/') + '_CLUSTERING'
    else:
        output_dir = args.input.rstrip('/') + '_CLUSTERING_FROM_' + '-'.join(args.source_clusters.split('/')[-6:-1])
    
    output_dir_feature = os.path.join(output_dir, f"feature={args.ext.split('.')[0]}")
    output_dir_distance = os.path.join(output_dir_feature, f'_distance={args.distance}')
    output_dir_nclusters = os.path.join(output_dir_distance, f'nclusters={args.num_clusters}')
    output_dir_path = output_dir_nclusters
    os.makedirs(output_dir_path, exist_ok=True)

    clusters_data = {}
    path_clusters_file = os.path.join(output_dir_path, f'clusters-data_feature={args.ext}_distance={args.distance}_nclusters={args.num_clusters}.pkl')


    # SEARCH FILES
    if not os.path.isfile(path_clusters_file):
        print(f'Searching files \'{args.ext}\' in \'{args.input}\'')
        files_paths = get_all_files_in_path(args.input, args.ext)
        print(f'Found {len(files_paths)} files\n------------------')

        clusters_data['files_paths'] = files_paths
        print(f'Saving found file paths to disk: \'{path_clusters_file}\'')
        save_dict(clusters_data, path_clusters_file)
    else:
        print(f'Loading saved paths: \'{path_clusters_file}\'')
        clusters_data = load_dict(path_clusters_file)
        files_paths   = clusters_data['files_paths']
        print(f'Loaded {len(files_paths)} files paths\n------------------')


    # LOAD FEATURES
    if not 'original_feats' in list(clusters_data.keys()):
        feat = torch.load(files_paths[0])
        # print('feat.shape:', feat.shape, '    feat.device:', feat.device)
        feat = torch.flatten(feat, start_dim=1)
        # print('feat.shape:', feat.shape, '    feat.dtype:', feat.dtype, '    feat.device:', feat.device)
        print(f"Allocating data matrix {(len(files_paths),feat.shape[1])}, device={torch.device('cpu')}...")
        all_feats = torch.zeros((len(files_paths),feat.shape[1]), dtype=torch.float, device=torch.device('cpu'))
        # print('all_feats.shape:', all_feats.shape, '    all_feats.dtype:', all_feats.dtype, '    all_feats.device:', all_feats.device)
        # print('------------------')
        # sys.exit(0)

        for idx_feat, file_path in enumerate(files_paths):
            print(f'Loading features {idx_feat}/{len(files_paths)} \'{file_path}\'          ', end='\r')
            feat = torch.load(file_path)
            feat = torch.flatten(feat, start_dim=1)
            all_feats[idx_feat,:] = feat
        print('')
        
        clusters_data['original_feats'] = all_feats
        print(f'Saving loaded features to disk: \'{path_clusters_file}\'')
        save_dict(clusters_data, path_clusters_file)
    else:
        print(f'Loading saved original features: \'{path_clusters_file}\'')
        clusters_data = load_dict(path_clusters_file)
        files_paths   = clusters_data['files_paths']
        all_feats     = clusters_data['original_feats'].cpu()
    print('all_feats.shape:', all_feats.shape, '    all_feats.dtype:', all_feats.dtype, '    all_feats.device:', all_feats.device)
    print('------------------')


    # RANDOM SELECTION
    if args.uniform_selection < 1.0:
        output_dir_path = os.path.join(output_dir_path, f'subsampl={args.uniform_selection}')
        os.makedirs(output_dir_path, exist_ok=True)

        name_clusters_file, ext_clusters_file = os.path.splitext(os.path.basename(path_clusters_file))
        fullname_subsampl_clusters_file = name_clusters_file + f'_subsampl={args.uniform_selection}' + ext_clusters_file
        path_clusters_file = os.path.join(output_dir_path, fullname_subsampl_clusters_file)

        if not os.path.isfile(path_clusters_file):
            rng = np.random.RandomState(440)
            indexes_used_samples = rng.choice(len(files_paths), round(len(files_paths)*args.uniform_selection), replace=False)
            print(f'    Randomly selecting samples {len(indexes_used_samples)}...')
            files_paths = [files_paths[idx] for idx in indexes_used_samples]
            all_feats   = all_feats[indexes_used_samples,:]

            clusters_data['files_paths']          = files_paths
            clusters_data['original_feats']       = all_feats
            clusters_data['indexes_used_samples'] = indexes_used_samples
            print(f'    Saving randomly selected file paths to disk: \'{path_clusters_file}\'')
            save_dict(clusters_data, path_clusters_file)
        else:
            print(f'    Loading randomly selected saved paths: \'{path_clusters_file}\'')
            clusters_data = load_dict(path_clusters_file)
            indexes_used_samples = clusters_data['indexes_used_samples']
            files_paths          = clusters_data['files_paths']
            all_feats            = clusters_data['original_feats'].cpu()
            print(f'    Loaded {len(files_paths)} files paths')
        print('    all_feats.shape:', all_feats.shape, '    all_feats.dtype:', all_feats.dtype, '    all_feats.device:', all_feats.device)
        print('------------------')


    # CLUSTERING
    if not args.source_clusters:
        # CLUSTERING ITSELF
        if not 'cluster_ids' in list(clusters_data.keys()) and not 'cluster_centers' in list(clusters_data.keys()):
            print(f'Clustering (K-Means)... num_clusters={args.num_clusters}')
            cluster_ids_x, cluster_centers = kmeans_pytorch.kmeans(X=all_feats,
                                                                   num_clusters=args.num_clusters,
                                                                   distance=args.distance,
                                                                   device=torch.device(args.device),
                                                                   seed=440)
            cluster_ids_x, cluster_centers = cluster_ids_x.cpu().numpy(), cluster_centers.cpu().numpy()
            # print('cluster_ids_x:', cluster_ids_x)
            print('cluster_ids_x.shape:', cluster_ids_x.shape)

            clusters_data['cluster_ids'] = cluster_ids_x
            clusters_data['cluster_centers'] = cluster_centers
            print(f'Saving clusters and predicted ids data to disk: \'{path_clusters_file}\'')
            save_dict(clusters_data, path_clusters_file)
        else:
            print(f'Loading saved clusters: \'{path_clusters_file}\'')
            cluster_ids_x   = clusters_data['cluster_ids']
            cluster_centers = clusters_data['cluster_centers']
        # print('cluster_ids_x.shape:', cluster_ids_x.shape, '    type(cluster_ids_x):', type(cluster_ids_x), '    cluster_ids_x.dtype:', cluster_ids_x.dtype)
        # print('cluster_centers.shape:', cluster_centers.shape, '    type(cluster_centers):', type(cluster_centers), '    cluster_centers.dtype:', cluster_centers.dtype)
    else:
        # LABELING USING CLUSTERS FROM OTHER DATASET
        print(f'Loading clusters from other dataset: \'{args.source_clusters}\'')
        clusters_other_dataset = load_dict(args.source_clusters)
        keep_keys = ['cluster_centers']
        for keep_key in keep_keys:
            assert keep_key in list(clusters_other_dataset.keys()), f"Key \'{keep_key}\' not found in file \'{args.source_clusters}\'"
        for found_key in list(clusters_other_dataset.keys()):
            if found_key != keep_key:
                del clusters_other_dataset[found_key]
        cluster_centers = clusters_other_dataset['cluster_centers']
        cluster_centers = torch.from_numpy(cluster_centers)

        print('Predicting samples ids from clusters...')
        cluster_ids_x = kmeans_pytorch.kmeans_predict(X=all_feats,
                                                      cluster_centers=cluster_centers,
                                                      distance=args.distance,
                                                      device=torch.device(args.device),
                                                      gamma_for_soft_dtw=0.001,
                                                      tqdm_flag=True)
        clusters_data['cluster_ids'] = cluster_ids_x
        print(f'Saving predicted ids data to disk: \'{path_clusters_file}\'')
        save_dict(clusters_data, path_clusters_file)
        # sys.exit(0)


    # DIMENSIONALITY REDUCTION
    if not 'feats_tsne' in list(clusters_data.keys()) and not 'cluster_centers_tsne' in list(clusters_data.keys()):
        print(f'Reducing dimensionality (TSNE)...')
        # all_feats_tsne = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(all_feats)  # for 'from tsnecuda import TSNE'
        all_feats_tsne = TSNE(n_components=2, metric=args.distance).fit_transform(np.vstack((all_feats.cpu().numpy(),cluster_centers)))    # from sklearn.manifold import TSNE
        cluster_centers_tsne = all_feats_tsne[len(files_paths):,:]
        all_feats_tsne       = all_feats_tsne[:len(files_paths),:]
        print('all_feats_tsne.shape:', all_feats_tsne.shape)

        clusters_data['feats_tsne'] = all_feats_tsne
        clusters_data['cluster_centers_tsne'] = cluster_centers_tsne
        print(f'Saving clusters (original and TSNE) data to disk: \'{path_clusters_file}\'')
        save_dict(clusters_data, path_clusters_file)
    else:
        print(f'Loading saved features and clusters TSNE: \'{path_clusters_file}\'')
        all_feats_tsne       = clusters_data['feats_tsne']
        cluster_centers_tsne = clusters_data['cluster_centers_tsne']

    chart_title = f'Face Style Clustering (num_clusters: {args.num_clusters}, distance: {args.distance})'
    chart_path = os.path.join(output_dir_path, f'clustering_feature={args.ext}_distance={args.distance}_nclusters={args.num_clusters}.png')
    print(f'Saving scatter plot of clusters: \'{chart_path}\'')
    save_scatter_plot_clusters(all_feats_tsne, cluster_ids_x, cluster_centers_tsne, chart_title, chart_path)


    # SAVE IMAGES OF CLUSTERS
    if args.num_imgs_clusters_to_save > 0:
        if not 'corresp_imgs_paths' in list(clusters_data.keys()):
            print(f'\nSearching corresponding images: \'{args.corresponding_imgs}\'')
            corresp_imgs_paths = [None] * len(files_paths)
            for idx_feat, file_path in enumerate(files_paths):
                # print(f'{idx_feat}/{len(files_paths)}')
                file_parent_dir = os.path.dirname(file_path)
                file_name = os.path.basename(file_path)

                img_parent_dir = file_parent_dir.replace(args.input, args.corresponding_imgs)
                img_name_pattern = img_parent_dir + '/' + file_name.replace(args.ext, '') + '.*'
                img_name_pattern = img_name_pattern.replace('[','*').replace(']','*')
                # print('img_name_pattern:', img_name_pattern)
                img_path = glob.glob(img_name_pattern)
                assert len(img_path) > 0, f'\nNo file found with the pattern \'{img_name_pattern}\''
                assert len(img_path) == 1, f'\nMore than 1 file found: \'{img_path}\''
                img_path = img_path[0]
                corresp_imgs_paths[idx_feat] = img_path
                # print('img_path:', img_path)
                print(f'{idx_feat}/{len(files_paths)} - img_path: \'{img_path}\'          ', end='\r')
                # sys.exit(0)
            print()
            assert len(corresp_imgs_paths) == len(files_paths)
            clusters_data['corresp_imgs_paths'] = corresp_imgs_paths
            print(f'Saving corresponding images paths to disk: \'{path_clusters_file}\'')
            save_dict(clusters_data, path_clusters_file)
        else:
            print(f'Loading saved corresponding images paths: \'{path_clusters_file}\'')
            corresp_imgs_paths = clusters_data['corresp_imgs_paths']

        output_dir_clusters_imgs = os.path.join(output_dir_path, f'clusters_imgs')
        os.makedirs(output_dir_clusters_imgs, exist_ok=True)

        print(f'\nCopying face images of clusters to: \'{output_dir_clusters_imgs}\'')
        for id_cluster_label, (cluster_label, src_img_path) in enumerate(zip(cluster_ids_x, corresp_imgs_paths)):
            if type(cluster_label) is torch.Tensor: cluster_label = cluster_label.item()
            output_dir_cluster = os.path.join(output_dir_clusters_imgs, str(cluster_label))
            os.makedirs(output_dir_cluster, exist_ok=True)
            print(f'{id_cluster_label}/{len(cluster_ids_x)}          ', end='\r')
            # print(f'{id_cluster_label}/{len(cluster_ids_x)} - \'{output_dir_cluster}\'          ', end='\r')
            
            # dst_img_path = os.path.join(output_dir_cluster, os.path.basename(src_img_path))
            dst_img_path = os.path.join(output_dir_cluster, src_img_path.split('/')[-2]+'_'+os.path.basename(src_img_path))
            shutil.copy(src_img_path, dst_img_path)
            # os.symlink(src_img_path, dst_img_path)

            # print()
            # sys.exit(0)
        print()


    # SEARCH FACIAL ATTRIBUTES FILES CONTAINING ETHNIC GROUPS PREDICTED LABELS
    if not 'facial_attribs_paths' in list(clusters_data.keys()):
        print(f'\nSearching corresponding facial attributes: \'{args.facial_attributes}\'')
        corresp_facial_attribs_paths = [None] * len(files_paths)
        for idx_file, file_path in enumerate(files_paths):
            file_parent_dir = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)

            attrib_parent_dir = file_parent_dir.replace(args.input, args.facial_attributes)
            attrib_name_pattern = attrib_parent_dir + '/' + file_name.replace(args.ext, '') + '.pkl'
            attrib_name_pattern = attrib_name_pattern.replace('[','*').replace(']','*')
            # print('attrib_name_pattern:', attrib_name_pattern)
            attrib_path = glob.glob(attrib_name_pattern)
            assert len(attrib_path) > 0, f'\nNo file found with the pattern \'{attrib_name_pattern}\''
            assert len(attrib_path) == 1, f'\nMore than 1 file found: \'{attrib_path}\''
            attrib_path = attrib_path[0]
            corresp_facial_attribs_paths[idx_file] = attrib_path
            # print('attrib_path:', attrib_path)
            print(f'{idx_file}/{len(files_paths)} - attrib_path: \'{attrib_path}\'          ', end='\r')
            # sys.exit(0)
        print()
        assert len(corresp_facial_attribs_paths) == len(files_paths)
        clusters_data['facial_attribs_paths'] = corresp_facial_attribs_paths
        print(f'Saving clusters data to disk: \'{path_clusters_file}\'')
        save_dict(clusters_data, path_clusters_file)
    else:
        print(f'\nLoading saved corresponding facial attributes paths: \'{path_clusters_file}\'')
        corresp_facial_attribs_paths = clusters_data['facial_attribs_paths']
    # sys.exit(0)


    # LOAD FACIAL ATTRIBUTES CONTAINING ETHNIC GROUPS LABELS
    if not 'facial_attribs' in list(clusters_data.keys()) and not 'dominant_races' in list(clusters_data.keys()):
        all_facial_attribs = [None] * len(corresp_facial_attribs_paths)
        all_dominant_races  = [None] * len(corresp_facial_attribs_paths)
        print(f'Loading corresponding individual facial attributes')
        for idx_file, attrib_path in enumerate(corresp_facial_attribs_paths):
            print(f'{idx_file}/{len(corresp_facial_attribs_paths)} - attrib_path: \'{attrib_path}\'          ', end='\r')
            facial_attribs = load_dict(attrib_path)
            all_facial_attribs[idx_file] = facial_attribs
            all_dominant_races[idx_file] = facial_attribs['race']['dominant_race']
        print()

        clusters_data['facial_attribs'] = all_facial_attribs
        clusters_data['dominant_races'] = all_dominant_races
        print(f'Saving clusters data to disk: \'{path_clusters_file}\'')
        save_dict(clusters_data, path_clusters_file)
    else:
        print(f'Loading saved corresponding facial attributes: \'{path_clusters_file}\'')
        all_facial_attribs = clusters_data['facial_attribs']
        all_dominant_races = clusters_data['dominant_races']


    # COUNT SAMPLES BELONGING TO EACH DISTINCT FACE STYLE (CLUSTER)
    races_labels_dict = {"asian": 0, "indian": 1, "black": 2, "white": 3, "middle eastern": 4, "latino hispanic": 5}
    if not 'races_styles_clusters_count' in list(clusters_data.keys()):
        races_styles_clusters_count = {race: np.zeros((args.num_clusters,)) for race in list(races_labels_dict.keys())}
        print(f'\nCounting face styles per race: {list(races_styles_clusters_count.keys())}')
        for idx_sample, (dominant_race, cluster_id) in enumerate(zip(all_dominant_races, cluster_ids_x)):
            print(f'{idx_sample}/{len(all_dominant_races)}', end='\r')
            races_styles_clusters_count[dominant_race][cluster_id] += 1
        print()
        # for idx_race, dominant_race in enumerate(list(races_styles_clusters_count.keys())):
        #     print(f'{dominant_race}:', races_styles_clusters_count[dominant_race], f'    type: {type(races_styles_clusters_count[dominant_race])}')
        clusters_data['races_styles_clusters_count'] = races_styles_clusters_count
        print(f'Saving clusters data to disk: \'{path_clusters_file}\'')
        save_dict(clusters_data, path_clusters_file)
    else:
        print(f'Loading saved face styles count: \'{path_clusters_file}\'')
        races_styles_clusters_count = clusters_data['races_styles_clusters_count']

    if not 'total_races' in list(races_styles_clusters_count.keys()):
        print(f'\nCounting total face styles...')
        races_styles_clusters_count_total_races = np.zeros((args.num_clusters,))
        for idx_race, race in enumerate(list(races_styles_clusters_count.keys())):
            races_styles_clusters_count_total_races += races_styles_clusters_count[race]
        races_styles_clusters_count['total_races'] = races_styles_clusters_count_total_races
        print('races_styles_clusters_count_total_races.sum():', races_styles_clusters_count_total_races.sum())
        # sys.exit(0)

    print('Normalizing races count...')
    races_styles_clusters_count_normalized = {}
    for idx_race, race in enumerate(list(races_styles_clusters_count.keys())):
        if races_styles_clusters_count[race].sum() > 0.0:
            races_styles_clusters_count_normalized[race] = races_styles_clusters_count[race] / races_styles_clusters_count[race].sum()
        else:
            races_styles_clusters_count_normalized[race] = np.zeros_like(races_styles_clusters_count[race])
        # print(f'{race}:{races_styles_clusters_count_normalized[race]}')



    print('Computing distributions statiscs...')
    races_styles_clusters_count_stats = {}
    for idx_race, race in enumerate(list(races_styles_clusters_count.keys())):
        races_styles_clusters_count_stats[race] = compute_statistical_metrics(races_styles_clusters_count_normalized[race])
        print(f'{race}: {races_styles_clusters_count_stats[race]}')



    global_title = 'Face Styles per Ethnic Group'
    styles_per_ethnic_group_path = os.path.join(output_dir_path, 'styles_per_ethnic_group.png')
    print(f'Saving chart of styles per race: \'{styles_per_ethnic_group_path}\'')
    # create_bar_chart(races_styles_clusters_count, global_title, styles_per_ethnic_group_path)
    save_styles_per_race_bars_chart(races_styles_clusters_count_normalized,
                                    races_styles_clusters_count_stats,
                                    global_title,
                                    styles_per_ethnic_group_path)



    title = 'Races Count'
    output_path = os.path.join(output_dir_path, 'races_count.png')
    # print(f'len(all_dominant_races): {len(all_dominant_races)}')
    print(f'Saving chart of races count: {output_path}')
    save_races_count_bar_chart(all_dominant_races, races_labels_dict, title, output_path)



    print('\nFinished!\n')


if __name__ == "__main__":
    args = parse_args()
    main(args)
