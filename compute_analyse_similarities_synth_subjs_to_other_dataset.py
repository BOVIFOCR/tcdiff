import os, sys
import numpy as np
import glob
from argparse import ArgumentParser
import time
import pickle


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--src_embedds_path', type=str, default='/datasets2/bjgbiesseck/face_recognition/synthetic/stylegan2-ada_SFHQ/3D_reconstruction_MICA/images/output/images')
    parser.add_argument('--src_embedds_ext', type=str, default='embedd_2D_arcface.npy')
    parser.add_argument('--tgt_embedds_path', type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/3D_reconstruction_MICA/real/1_CASIA-WebFace/imgs_crops_112x112/embeddings2D')
    parser.add_argument('--tgt_embedds_ext', type=str, default='_mean_embedding_r100_arcface.npy')

    parser.add_argument('--output_path', type=str, default='/datasets2/bjgbiesseck/face_recognition/synthetic/stylegan2-ada_SFHQ/images_selected_different_from_CASIA-WebFace')
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


def load_embedds(files_paths, normalize=True):
    embedds_dict = {}
    for idx_path, file_path in enumerate(files_paths):
        subj_name = file_path.split('/')[-2]
        print(f'    {idx_path}/{len(files_paths)} - subj: {subj_name}', end='\r')
        embedd = np.load(file_path)
        if normalize:
            norm = np.linalg.norm(embedd)
            embedd /= norm
        embedds_dict[subj_name] = embedd
    print('')
    return embedds_dict


def cosine_similarity(A, B):
    A = np.squeeze(A)
    B = np.squeeze(B)
    cos_sim = np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))
    return cos_sim


def compute_all_cosine_similarities(src_embedds, tgt_embedds):
    all_cos_src_similarities = {}
    total_time = 0.0
    for idx_src_subj, src_subj in enumerate(src_embedds.keys()):
        start_time = time.time()
        src_embedd = src_embedds[src_subj]
        subj_sims = {}
        for idx_tgt_subj, tgt_subj in enumerate(tgt_embedds.keys()):
            if src_subj != tgt_subj:
                tgt_embedd = tgt_embedds[tgt_subj]
                cos_sim_src_tgt = cosine_similarity(src_embedd, tgt_embedd)
                subj_sims[tgt_subj] = cos_sim_src_tgt
                print(f'    src_subj: {idx_src_subj}/{len(src_embedds)}  -  tgt_subj: {idx_tgt_subj}/{len(tgt_embedds)}  -  cos_sim: {cos_sim_src_tgt}                          ', end='\r')
                
        all_cos_src_similarities[src_subj] = subj_sims
        print('')
        end_time = time.time()
        spent_time = end_time - start_time
        total_time += spent_time
        est_time = spent_time * (len(src_embedds.keys())-idx_src_subj)
        print('        Elapsed time: %.2fs  -  Total Elapsed time: %.2fs (%.2fh)  -  Time to end: %.2fs (%.2fh)' % (spent_time, total_time, total_time/3600, est_time, est_time/3600))
    print('')
    return all_cos_src_similarities


def compute_save_all_cosine_similarities(src_embedds, tgt_embedds, output_path):
    total_time = 0.0
    for idx_src_subj, src_subj in enumerate(src_embedds.keys()):
        start_time = time.time()
        src_embedd = src_embedds[src_subj]
        subj_sims = {}
        for idx_tgt_subj, tgt_subj in enumerate(tgt_embedds.keys()):
            if src_subj != tgt_subj:
                tgt_embedd = tgt_embedds[tgt_subj]
                cos_sim_src_tgt = cosine_similarity(src_embedd, tgt_embedd)
                subj_sims[tgt_subj] = cos_sim_src_tgt
                print(f'    src_subj: {idx_src_subj}/{len(src_embedds)}  -  tgt_subj: {idx_tgt_subj}/{len(tgt_embedds)}  -  cos_sim: {cos_sim_src_tgt}                          ', end='\r')
        print('')

        path_subj = os.path.join(output_path, src_subj)
        os.makedirs(path_subj, exist_ok=True)
        path_file_cos_sims = os.path.join(path_subj, 'cosine_similarities.pkl')
        print(f'        Saving cosine similarities: {path_file_cos_sims}')
        save_object_pickle(subj_sims, path_file_cos_sims)
        
        end_time = time.time()
        spent_time = end_time - start_time
        total_time += spent_time
        est_time = spent_time * (len(src_embedds.keys())-idx_src_subj)
        print('        Elapsed time: %.2fs  -  Total Elapsed time: %.2fs (%.2fh)  -  Time to end: %.2fs (%.2fh)' % (spent_time, total_time, total_time/3600, est_time, est_time/3600))



def main(args):
    args.src_embedds_path = args.src_embedds_path.rstrip('/')
    args.tgt_embedds_path = args.tgt_embedds_path.rstrip('/')
    assert os.path.exists(args.src_embedds_path), f'Error, no such directory \'{args.src_embedds_path}\''
    assert os.path.exists(args.tgt_embedds_path), f'Error, no such directory \'{args.tgt_embedds_path}\''

    output_path = args.output_path.rstrip('/')
    os.makedirs(output_path, exist_ok=True)

    imgs_output_path = os.path.join(output_path, 'samples')
    os.makedirs(imgs_output_path, exist_ok=True)

    cossim_output_path = os.path.join(output_path, 'cosine_similarities')
    os.makedirs(cossim_output_path, exist_ok=True)



    print(f'Loading files \'{args.src_embedds_ext}\' in path: \'{args.src_embedds_path}\'')
    all_src_embedds_path = find_all_files(args.src_embedds_path, [args.src_embedds_ext])

    print(f'    Loaded {len(all_src_embedds_path)} files')

    print(f'Loading files \'{args.tgt_embedds_ext}\' in path: \'{args.tgt_embedds_path}\'')
    all_tgt_embedds_path = find_all_files(args.tgt_embedds_path, [args.tgt_embedds_ext])

    print(f'    Loaded {len(all_tgt_embedds_path)} files')


    
    print(f'\nLoading source embeddings')
    all_src_embedds = load_embedds(all_src_embedds_path, normalize=True)
    print(f'    Loaded {len(all_src_embedds)} source embeddings')

    print(f'\nLoading target embeddings')
    all_tgt_embedds = load_embedds(all_tgt_embedds_path, normalize=True)
    print(f'    Loaded {len(all_tgt_embedds)} target embeddings')



    print(f'\nComputing cosine similarities')
    compute_save_all_cosine_similarities(all_src_embedds, all_tgt_embedds, cossim_output_path)

    print('\nFinished\n')

   


if __name__ == "__main__":
    args = parse_args()
    main(args)