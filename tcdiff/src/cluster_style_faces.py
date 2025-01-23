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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_crops_112x112_JUST-PROTOCOL-IMGS_STYLE_FEATURES')
    parser.add_argument('--ext', type=str, default='_spatial.pt')
    parser.add_argument('--corresponding-imgs', type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_crops_112x112_JUST-PROTOCOL-IMGS', help='')
    parser.add_argument('--num-imgs-clusters-to-save', type=int, default=0)
    parser.add_argument('--num-clusters', type=int, default=100)
    parser.add_argument('--distance', type=str, default='all', help='all or euclidean or cosine')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda:0 or cpu')
    parser.add_argument('--facial-attributes', type=str, default='', help='')    # '/datasets2/1st_frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_crops_112x112_JUST-PROTOCOL-IMGS_FACE_ATTRIB'
    parser.add_argument('--source-clusters', type=str, default='', help='')    # '/datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112_STYLE_FEATURES_CLUSTERING/feature=_style/_distance=cosine/nclusters=100/clusters-data_feature=_style.pt_distance=cosine_nclusters=100.pkl'


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


def save_styles_per_race_bars_chart(ndarrays, global_title, output_path):
    races = list(ndarrays.keys())
    ndarrays = [ndarrays[race] for race in races]
    if len(ndarrays) != len(races):
        raise ValueError("The number of ndarrays must match the number of subtitles.")

    # global_max = max([arr.max() for arr in ndarrays])
    global_max = 0.1   # 10%

    n_subplots = len(ndarrays)
    fig_height = 10
    fig, axes = plt.subplots(n_subplots, 1, figsize=(10, fig_height), constrained_layout=True)

    if n_subplots == 1:
        axes = [axes]

    fig.suptitle(global_title, fontsize=16, weight='bold')
    for i, (ax, arr, subtitle) in enumerate(zip(axes, ndarrays, races)):
        ax.bar(range(len(arr)), arr)
        ax.set_ylim(0, global_max)
        ax.set_yticks([0, global_max])
        ax.set_title(subtitle, fontsize=14)
        if i == len(ndarrays)-1: ax.set_xlabel("Face Styles", fontsize=12)
        ax.set_ylabel("Percentual", fontsize=12)

        ax.set_xticks(range(len(arr)))
        ax.set_xticklabels(range(len(arr)), fontsize=8, rotation=90)

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

    # SEARCH FILES AND LOAD FEATURES
    if not os.path.isfile(path_clusters_file):
        print(f'Searching files \'{args.ext}\' in \'{args.input}\'')
        files_paths = get_all_files_in_path(args.input, args.ext)
        print(f'Found {len(files_paths)} files\n------------------')

        clusters_data['files_paths'] = files_paths
        print(f'Saving found file paths to disk: \'{path_clusters_file}\'')
        save_dict(clusters_data, path_clusters_file)

        feat = torch.load(files_paths[0])
        # print('feat.shape:', feat.shape, '    feat.device:', feat.device)
        feat = torch.flatten(feat, start_dim=1)
        # print('feat.shape:', feat.shape, '    feat.dtype:', feat.dtype, '    feat.device:', feat.device)
        print(f'Allocating data matrix {(len(files_paths),feat.shape[1])}...')
        all_feats = torch.zeros((len(files_paths),feat.shape[1]), dtype=torch.float, device='cuda:0')
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
        print(f'Saving found file paths and loaded features to disk: \'{path_clusters_file}\'')
        save_dict(clusters_data, path_clusters_file)
    else:
        print(f'Loading saved files paths and original features: \'{path_clusters_file}\'')
        clusters_data = load_dict(path_clusters_file)
        files_paths   = clusters_data['files_paths']
        all_feats     = clusters_data['original_feats'].cpu()
    print('all_feats.shape:', all_feats.shape, '    all_feats.dtype:', all_feats.dtype, '    all_feats.device:', all_feats.device)
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
            output_dir_cluster = os.path.join(output_dir_clusters_imgs, str(cluster_label))
            os.makedirs(output_dir_cluster, exist_ok=True)
            print(f'{id_cluster_label}/{len(cluster_ids_x)}          ', end='\r')
            # print(f'{id_cluster_label}/{len(cluster_ids_x)} - \'{output_dir_cluster}\'          ', end='\r')
            
            dst_img_path = os.path.join(output_dir_cluster, os.path.basename(src_img_path))
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
        print(f'\nCounting face styles per race: {list(races_labels_dict.keys())}')
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

    print('Normalizing races count...')
    races_styles_clusters_count_normalized = {}
    for idx_race, race in enumerate(list(races_labels_dict.keys())):
        if races_styles_clusters_count[race].sum() > 0.0:
            races_styles_clusters_count_normalized[race] = races_styles_clusters_count[race] / races_styles_clusters_count[race].sum()
        else:
            races_styles_clusters_count_normalized[race] = np.zeros_like(races_styles_clusters_count[race])
        # print(f'{race}:{races_styles_clusters_count_normalized[race]}')



    global_title = 'Face Styles per Ethnic Group'
    styles_per_ethnic_group_path = os.path.join(output_dir_path, 'styles_per_ethnic_group.png')
    print(f'Saving chart of styles per race: \'{styles_per_ethnic_group_path}\'')
    # create_bar_chart(races_styles_clusters_count, global_title, styles_per_ethnic_group_path)
    save_styles_per_race_bars_chart(races_styles_clusters_count_normalized, global_title, styles_per_ethnic_group_path)



    title = 'Races Count'
    output_path = os.path.join(output_dir_path, 'races_count.png')
    # print(f'len(all_dominant_races): {len(all_dominant_races)}')
    print(f'Saving chart of races count: {output_path}')
    save_races_count_bar_chart(all_dominant_races, races_labels_dict, title, output_path)



    print('\nFinished!\n')


if __name__ == "__main__":
    args = parse_args()
    main(args)