import os, sys
from argparse import ArgumentParser
import re

import torch
import numpy as np
import glob
import shutil
import pickle
from kmeans_pytorch import kmeans
import tsnecuda
# from tsnecuda import TSNE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112_STYLE_FEATURES_TINY')
    parser.add_argument('--ext', type=str, default='_spatial.pt')
    parser.add_argument('--corresponding-imgs', type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112', help='')
    parser.add_argument('--num-imgs-clusters-to-save', type=int, default=0)
    parser.add_argument('--num-clusters', type=int, default=100)
    parser.add_argument('--distance', type=str, default='all', help='all or euclidean or cosine')
    parser.add_argument('--facial-attributes', type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112_FACE_ATTRIB', help='')

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


def main(args):

    output_dir = args.input.rstrip('/') + '_CLUSTERING'
    output_dir_feature = os.path.join(output_dir, f"feature={args.ext.split('.')[0]}")
    output_dir_distance = os.path.join(output_dir_feature, f'_distance={args.distance}')
    os.makedirs(output_dir_distance, exist_ok=True)

    clusters_data = {}
    path_clusters_file = os.path.join(output_dir_distance, f'clusters-data_feature={args.ext}_distance={args.distance}_nclusters={args.num_clusters}.pkl')

    if not os.path.isfile(path_clusters_file):
        print(f'Searching files \'{args.ext}\' in \'{args.input}\'')
        files_paths = get_all_files_in_path(args.input, args.ext)
        print(f'Found {len(files_paths)} files\n------------------')

        clusters_data['files_paths'] = files_paths
        print(f'Saving clusters data to disk: \'{path_clusters_file}\'')
        save_dict(clusters_data, path_clusters_file)


        feat = torch.load(files_paths[0])
        # print('feat.shape:', feat.shape, '    feat.device:', feat.device)
        feat = torch.flatten(feat, start_dim=1)
        # print('feat.shape:', feat.shape, '    feat.dtype:', feat.dtype, '    feat.device:', feat.device)
        print(f'Allocating data matrix {(len(files_paths),feat.shape[1])}...')
        all_feats = torch.zeros((len(files_paths),feat.shape[1]), dtype=torch.float, device='cuda:0')
        print('all_feats.shape:', all_feats.shape, '    all_feats.dtype:', all_feats.dtype, '    all_feats.device:', all_feats.device)
        print('------------------')
        # sys.exit(0)

        for idx_feat, file_path in enumerate(files_paths):
            print(f'Loading features {idx_feat}/{len(files_paths)} \'{file_path}\'          ', end='\r')
            feat = torch.load(file_path)
            feat = torch.flatten(feat, start_dim=1)
            all_feats[idx_feat,:] = feat
        print('')
        
        clusters_data['original_feats'] = all_feats
        print(f'Saving clusters data to disk: \'{path_clusters_file}\'')
        save_dict(clusters_data, path_clusters_file)

    else:
        print(f'Loading clusters from disk: \'{path_clusters_file}\'')
        clusters_data = load_dict(path_clusters_file)
        files_paths   = clusters_data['files_paths']
        all_feats     = clusters_data['original_feats']



    if not 'cluster_ids' in list(clusters_data.keys()) and not 'cluster_centers' in list(clusters_data.keys()):
        print(f'Clustering (K-Means)... num_clusters={args.num_clusters}')
        cluster_ids_x, cluster_centers = kmeans(X=all_feats,
                                                num_clusters=args.num_clusters,
                                                distance=args.distance,
                                                device=torch.device('cuda:0'),
                                                seed=440)
        cluster_ids_x, cluster_centers = cluster_ids_x.cpu().numpy(), cluster_centers.cpu().numpy()
        # print('cluster_ids_x:', cluster_ids_x)
        print('cluster_ids_x.shape:', cluster_ids_x.shape)

        clusters_data['cluster_ids'] = cluster_ids_x
        clusters_data['cluster_centers'] = cluster_centers
        print(f'Saving clusters data to disk: \'{path_clusters_file}\'')
        save_dict(clusters_data, path_clusters_file)

    else:
        cluster_ids_x   = clusters_data['cluster_ids']
        cluster_centers = clusters_data['cluster_centers']



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
        print(f'Loading features and clusters TSNE: \'{path_clusters_file}\'')
        all_feats_tsne       = clusters_data['feats_tsne']
        cluster_centers_tsne = clusters_data['cluster_centers_tsne']

    chart_title = f'Face Style Clustering (num_clusters: {args.num_clusters}, distance: {args.distance})'
    chart_path = os.path.join(output_dir_distance, f'clustering_feature={args.ext}_distance={args.distance}_nclusters={args.num_clusters}.png')
    print(f'Saving scatter plot of clusters: \'{chart_path}\'')
    save_scatter_plot_clusters(all_feats_tsne, cluster_ids_x, cluster_centers_tsne, chart_title, chart_path)



    if args.num_imgs_clusters_to_save > 0:
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

        output_dir_clusters_imgs = os.path.join(output_dir_distance, f'clusters_imgs_nclusters={args.num_clusters}')
        os.makedirs(output_dir_clusters_imgs, exist_ok=True)

        print(f'\nCopying face images to: \'{output_dir_clusters_imgs}\'')
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

        attrib = load_dict(attrib_path)
        print("\nattrib['race']:", attrib)
        sys.exit(0)

    print()
    assert len(corresp_facial_attribs_paths) == len(files_paths)

    print('\nFinished!\n')


if __name__ == "__main__":
    args = parse_args()
    main(args)