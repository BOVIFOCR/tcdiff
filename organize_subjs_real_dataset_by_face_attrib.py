import os, sys
import numpy as np
import glob
from argparse import ArgumentParser
import time
import pickle
import matplotlib.pyplot as plt


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--face_attrib_path', type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112_FACE_ATTRIB')
    parser.add_argument('--face_attrib_ext', type=str, default='.pkl')
    parser.add_argument('--imgs_path', type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/real/1_CASIA-WebFace/imgs_crops_112x112')
    parser.add_argument('--imgs_ext', type=str, default='.png')
    parser.add_argument('--output_path', type=str, default='')

    args = parser.parse_args()
    return args


def save_object_pickle(obj, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)


def load_object_pickle(file_path):
    with open(file_path, 'rb') as file:
        obj = pickle.load(file)
    return obj


def find_all_files(folder_path, extensions=['.jpg', '.png']):
    image_paths = []
    num_found_files = 0
    for root, _, files in os.walk(folder_path):
        for ext in extensions:
            pattern = os.path.join(root, '*' + ext)
            matching_files = glob.glob(pattern)
            image_paths.extend(matching_files)
            num_found_files += 1
            print(f'    num_found_files: {num_found_files}', end='\r')
    print('')
    return sorted(image_paths)


def load_files_data(files_paths):
    cossims_dict = {}
    for idx_path, file_path in enumerate(files_paths):
        subj_name = file_path.split('/')[-2]
        print(f'    {idx_path}/{len(files_paths)} - subj: {subj_name}', end='\r')
        cossims = load_object_pickle(file_path)
        cossims_dict[subj_name] = cossims
    print('')
    return cossims_dict


def merge_similarities(all_cossim_subj):
    subjs_src = sorted(list(all_cossim_subj.keys()))
    num_subj_src = len(subjs_src)
    num_subj_tgt = len(all_cossim_subj[subjs_src[0]].keys())
    all_sims_merged = np.zeros((num_subj_src*num_subj_tgt,), dtype=np.float32)
    idx_all_sims = 0
    for idx_subj_src, subj_src in enumerate(subjs_src):
        sims_subj_src = all_cossim_subj[subj_src]
        subjs_tgt = sorted(list(sims_subj_src.keys()))
        for idx_subj_tgt, subj_tgt in enumerate(subjs_tgt):
            print(f'    subj_src: {idx_subj_src}/{len(subjs_src)} ({subj_src})    subj_tgt: {idx_subj_tgt}/{len(subjs_tgt)} ({subj_tgt})          ', end='\r')
            cossim_subj_src_to_tgt = sims_subj_src[subj_tgt]
            all_sims_merged[idx_all_sims] = cossim_subj_src_to_tgt
            idx_all_sims += 1
    print('')
    return all_sims_merged


def save_histograms(final_hist_cossim, final_bin_edges, min_cossim, max_cossim, filename, title):

    final_hist_cossim /= np.sum(final_hist_cossim)
    bin_width = final_bin_edges[1] - final_bin_edges[0]
    label = 'All cos. similarities (min=%.2f, max=%.2f)' % (min_cossim, max_cossim)
    plt.bar(final_bin_edges[:-1], final_hist_cossim, width=bin_width, alpha=0.7, label=label)










    plt.title(title)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.legend()

    plt.xlim([-1, 1])
    plt.ylim([0, 1.0])
    plt.savefig(filename)




def main(args):
    args.face_attrib_path = args.face_attrib_path.rstrip('/')
    assert os.path.exists(args.face_attrib_path), f'Error, no such directory \'{args.face_attrib_path}\''
    args.imgs_path = args.imgs_path.rstrip('/')
    assert os.path.exists(args.imgs_path), f'Error, no such directory \'{args.imgs_path}\''

    if args.output_path == '':
        output_path = args.imgs_path + '_ORGANIZED_BY_FACE_ATTRIBS'
    os.makedirs(output_path, exist_ok=True)



    print(f'Searching face attribs files \'{args.face_attrib_ext}\' in path: \'{args.face_attrib_path}\'')
    attrib_path_list = find_all_files(args.face_attrib_path, [args.face_attrib_ext])
    print(f'    Loaded {len(attrib_path_list)} files\n')


    all_attribs_list = [None] * len(attrib_path_list)
    for i, attrib_path in enumerate(attrib_path_list):
        print(f'Loading attributes files - {i}/{len(attrib_path_list)} - \'{attrib_path}\'                ', end='\r')
        all_attribs_list[i] = load_object_pickle(attrib_path)
    print('')




    print(f'\nFiltering subjects')
    num_selected_subj, num_rejected_subj = 0, 0
    for idx_cossim, cossim_path in enumerate(all_cossim_path):
        subj_name = cossim_path.split('/')[-2]
        subj_sample_pattern = os.path.join(args.samples_path, f"{subj_name}*.{args.samples_ext.lstrip('.')}")
        subj_sample_path = glob.glob(subj_sample_pattern)
        assert len(subj_sample_path) > 0, f'Error, no such file with pattern \'{subj_sample_pattern}\''
        subj_sample_path = subj_sample_path[0]

        cossims = load_object_pickle(cossim_path)
        
        reject_subj = False
        for idx_key, subj_key in enumerate(cossims.keys()):
            if cossims[subj_key] > args.thresh:
                reject_subj = True
                break
        
        try:
            if reject_subj:
                num_rejected_subj += 1
                os.symlink(subj_sample_path, os.path.join(rejected_samples_dir, os.path.basename(subj_sample_path)))
            else:
                num_selected_subj += 1
                os.symlink(subj_sample_path, os.path.join(selected_samples_dir, os.path.basename(subj_sample_path)))
        except FileExistsError:
            pass
        
        print(f'    {idx_cossim}/{len(all_cossim_path)} - subj: {subj_name} - reject_subj: {reject_subj} - num_rejected_subj: {num_rejected_subj} - num_selected_subj: {num_selected_subj}')

    print('\nFinished\n')

   


if __name__ == "__main__":
    args = parse_args()
    main(args)