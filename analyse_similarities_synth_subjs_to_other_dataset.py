import os, sys
import numpy as np
import glob
from argparse import ArgumentParser
import time
import pickle
import matplotlib.pyplot as plt


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--cos_sim_path', type=str, default='/datasets2/bjgbiesseck/face_recognition/synthetic/stylegan2-ada_SFHQ/a_tiny_sample_130_images_selected_different_from_CASIA-WebFace/cosine_similarities')
    parser.add_argument('--cos_sim_ext', type=str, default='cosine_similarities.pkl')
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--num_parts_data', type=int, default=10)
    args = parser.parse_args()
    return args


def divide_list_into_parts(list_size, num_parts):
    part_size = list_size // num_parts
    remain = list_size % num_parts
    begin_indexes = []
    end_indexes = []

    start = 0
    for i in range(num_parts):
        end = min(start+part_size, list_size)
        begin_indexes.append(start)
        end_indexes.append(end)
        start = end
    end_indexes[-1] += remain

    num_elements = sum([end_indexes[i]-begin_indexes[i] for i in range(num_parts)])
    assert num_elements == list_size, f'Error, num_elements ({num_elements}) != list_size ({list_size}). They must be equal!'

    return begin_indexes, end_indexes


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
    args.cos_sim_path = args.cos_sim_path.rstrip('/')
    assert os.path.exists(args.cos_sim_path), f'Error, no such directory \'{args.cos_sim_path}\''

    output_path = os.path.dirname(args.cos_sim_path)
    


    print(f'Loading files \'{args.cos_sim_ext}\' in path: \'{args.cos_sim_path}\'')
    all_cossim_path = find_all_files(args.cos_sim_path, [args.cos_sim_ext])
    print(f'    Loaded {len(all_cossim_path)} files')


    begin_indexes, end_indexes = divide_list_into_parts(len(all_cossim_path), args.num_parts_data)




    hist_bins = np.arange(-1.0, 1.1, 0.025)
    final_hist_cossim = np.zeros((len(hist_bins)-1,), dtype=np.float32)
    min_cossim, max_cossim = 1, -1

    for idx_part, (begin_idx, end_idx) in enumerate(zip(begin_indexes, end_indexes)):
        cossim_path_part = all_cossim_path[begin_idx:end_idx]
        print(f'\nLoading similarities - part {idx_part}/{args.num_parts_data} - begin_idx {begin_idx} - end_idx {end_idx} - num_files {end_idx-begin_idx}')
        cossim_subj_part = load_files_data(cossim_path_part)
        print(f'    Loaded {len(cossim_subj_part)} files')

        print(f'    Merging similarities')
        cossim_merged_part = merge_similarities(cossim_subj_part)
        if cossim_merged_part.min() < min_cossim: min_cossim = cossim_merged_part.min()
        if cossim_merged_part.max() > max_cossim: max_cossim = cossim_merged_part.max()
        hist_cossim_part, bin_cossim_edges = np.histogram(cossim_merged_part, bins=hist_bins)
        final_hist_cossim += hist_cossim_part

    hist_file_name = 'hist_cosine_similarities.png'
    hist_file_path = os.path.join(output_path, hist_file_name)
    title = 'Cosine similarities'
    print(f'\nSaving histogram: \'{hist_file_path}\'')
    save_histograms(final_hist_cossim, hist_bins, min_cossim, max_cossim, hist_file_path, title)

    print('\nFinished\n')

   


if __name__ == "__main__":
    args = parse_args()
    main(args)