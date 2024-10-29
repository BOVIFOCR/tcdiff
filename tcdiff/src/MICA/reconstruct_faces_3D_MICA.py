
















import argparse
import os, sys
import random
from glob import glob
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import face_align
from loguru import logger
from pytorch3d.io import save_ply, save_obj
from skimage.io import imread
from tqdm import tqdm

from configs.config import get_cfg_defaults
from datasets.creation.util import get_arcface_input, get_center
from utils import util


def deterministic(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    np.random.seed(rank)
    random.seed(rank)

    cudnn.deterministic = True
    cudnn.benchmark = False


def adjust_output_folders_names(args):
    if args.i.split('/')[-1] != args.o.split('/')[-1]:
        args.o = os.path.join(args.o, args.i.split('/')[-1])
    args.o = os.path.join(args.o, 'output')

    if args.a == '':
        args.a = os.path.join('/'.join(args.o.split('/')[:-1]), 'arcface')


def add_line_to_text_file(path_file, line):
    with open(path_file, 'a') as file:
        file.write(line + '\n')


def process_BERNARDO(sub_folder, args, app, ignore_face_not_det, path_file_faces_not_det, image_size=224):
    dst = Path(args.a)
    dst.mkdir(parents=True, exist_ok=True)

    processes = []
    image_paths = sorted(glob(sub_folder + '/*.jpg')) + sorted(glob(sub_folder + '/*.png'))
    for image_path in tqdm(image_paths):
        if args.str_pattern in image_path:
            args.str_pattern = ''
            img = cv2.imread(image_path)

            bboxes, kpss = app.det_model.detect(img, max_num=0, metric='default')
            if bboxes.shape[0] == 0:
                if ignore_face_not_det:
                    add_line_to_text_file(path_file_faces_not_det, image_path)
                    continue
                else:
                    print(f'Error: face not detected in image \'{image_path}\'\n')
                    sys.exit(0)
            i = get_center(bboxes, img)
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]

            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            blob, aimg = get_arcface_input(face, img)

            path_arcface_img = image_path.replace(args.i, args.a)
            Path(os.path.dirname(path_arcface_img)).mkdir(exist_ok=True, parents=True)
            path_arcface_img = path_arcface_img.replace('.png', '.jpg').replace('.jpeg', '.jpg')
            path_arcface_npy = path_arcface_img.replace('.jpg', '.npy')

            cv2.imwrite(path_arcface_img, aimg)
            np.save(path_arcface_npy, blob)
            processes.append(path_arcface_npy)

            if args.save_lmk:
                img_copy = img.copy()
                for kp in kps:
                    x, y = kp
                    cv2.circle(img_copy, (int(x), int(y)), 1, (0, 0, 255), 1)
                path_face_with_lmk = path_arcface_img.replace('.jpg', '_lmk.jpg')
                cv2.imwrite(path_face_with_lmk, img_copy)



    return processes


def process_BERNARDO_no_face_det(sub_folder, args, app, image_size=224):
    dst = Path(args.a)
    dst.mkdir(parents=True, exist_ok=True)

    processes = []
    image_paths = sorted(glob(sub_folder + '/*.jpg')) + sorted(glob(sub_folder + '/*.png'))
    for image_path in tqdm(image_paths):
        if args.str_pattern in image_path:
            args.str_pattern = ''
            img = cv2.imread(image_path)

            bbox = [0., 0., img.shape[0], img.shape[1]]
            kps = np.array(
                    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                    [41.5493, 92.3655], [70.7299, 92.2041]],
                    dtype=np.float32)
            det_score = 0.99

            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            blob, aimg = get_arcface_input(face, img)

            path_arcface_img = image_path.replace(args.i, args.a)
            Path(os.path.dirname(path_arcface_img)).mkdir(exist_ok=True, parents=True)
            path_arcface_img = path_arcface_img.replace('.png', '.jpg').replace('.jpeg', '.jpg')
            path_arcface_npy = path_arcface_img.replace('.jpg', '.npy')

            cv2.imwrite(path_arcface_img, aimg)
            np.save(path_arcface_npy, blob)
            processes.append(path_arcface_npy)


    return processes



def to_batch(path):
    src = path.replace('.npy', '.jpg')
    if not os.path.exists(src):
        src = path.replace('.npy', '.png')

    image = imread(src)[:, :, :3]
    image = image / 255.
    image = cv2.resize(image, (224, 224)).transpose(2, 0, 1)
    image = torch.tensor(image).cuda()[None]

    arcface = np.load(path)
    arcface = torch.tensor(arcface).cuda()[None]

    return image, arcface


def load_checkpoint(args, mica):
    checkpoint = torch.load(args.m)
    if 'arcface' in checkpoint:
        mica.arcface.load_state_dict(checkpoint['arcface'])
    if 'flameModel' in checkpoint:
        mica.flameModel.load_state_dict(checkpoint['flameModel'])


def get_parts_indices(sub_folders, divisions):
    begin_div = []
    end_div = []
    div_size = int(len(sub_folders) / divisions)
    remainder = int(len(sub_folders) % divisions)

    for i in range(0, divisions):
        begin_div.append(i*div_size)
        end_div.append(i*div_size + div_size)
    
    end_div[-1] += remainder




    return begin_div, end_div


def filter_points_in_sphere(pointcloud, point, radius):
    if torch.is_tensor(pointcloud):
        squared_distances = torch.sum((pointcloud - point) ** 2, axis=1)
        indices_inside_sphere = squared_distances <= radius**2
    elif isinstance(pointcloud, np.ndarray):
        squared_distances = np.sum((pointcloud - point) ** 2, axis=1)
        indices_inside_sphere = squared_distances <= radius**2

    points_inside_sphere = pointcloud[indices_inside_sphere]
    return points_inside_sphere


def translate_point_cloud(pointcloud, point):
    pointcloud = pointcloud - point
    return pointcloud


def get_all_paths_from_file(file_path, pattern=''):
    with open(file_path, 'r') as file:
        all_lines = [line.strip() for line in file.readlines()]
        valid_lines = []
        for i, line in enumerate(all_lines):
            if pattern in line:
                valid_lines.append(line)
        valid_lines.sort()
        return valid_lines


def get_all_subfolders_from_file_paths(file_path, pattern=''):
    file_paths = get_all_paths_from_file(file_path, pattern)
    subfolders = [None] * len(file_paths)
    for i, file_path in enumerate(file_paths):
        subfolder = '/'.join(file_path.split('/')[:-1])
        subfolders[i] = subfolder
    subfolders = list(set(subfolders))
    subfolders.sort()
    return subfolders


class Tree:
    def walk(self, dir_path: Path):
        contents = list(dir_path.iterdir())
        for path in contents:
            if path.is_dir():  # extend the prefix and recurse:
                yield str(path)
                yield from self.walk(path)

    def get_all_sub_folders(self, dir_path: str):
        folders = [dir_path]
        for folder in Tree().walk(Path(os.getcwd()) / dir_path):

            folders.append(folder)
        return sorted(folders)



def main(cfg, args):
    assert args.part < args.div, f'Error, args.part ({args.part}) >= args.div ({args.div}). args.part must be less than args.div!'

    device = 'cuda:0'

    cfg.model.testing = True
    mica = util.find_model_using_name(model_dir='micalib.models', model_name=cfg.model.name)(cfg, device)
    load_checkpoint(args, mica)
    mica.eval()

    faces = mica.render.faces[0].cpu()
    Path(args.o).mkdir(exist_ok=True, parents=True)

    app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_thresh=0.1, det_size=(224, 224))

    with torch.no_grad():
        logger.info(f'Processing has started...')

        if args.file_list != '' and os.path.isfile(args.file_list):
            print(f'\nLoading subfolders with pattern \'{args.str_pattern}\' from file \'{args.file_list}\' ...')
            all_sub_folders = get_all_subfolders_from_file_paths(args.file_list, args.str_pattern)
        else:
            print('\nLoading subfolders paths...')
            all_sub_folders = Tree().get_all_sub_folders(args.i)

        begin_parts, end_parts = get_parts_indices(all_sub_folders, args.div)
        sub_folders = all_sub_folders[begin_parts[args.part]:end_parts[args.part]]

        begin_index_str = 0
        end_index_str = len(sub_folders)

        if args.str_begin != '':
            print('Searching str_begin \'' + args.str_begin + '\' ...  ')
            for x, sub_folder in enumerate(sub_folders):
                if args.str_begin in sub_folder:
                    begin_index_str = x
                    print('found at', begin_index_str)
                    break

        if args.str_end != '':
            print('Searching str_end \'' + args.str_end + '\' ...  ')
            for x, sub_folder in enumerate(sub_folders):
                if args.str_end in sub_folder:
                    end_index_str = x+1
                    print('found at', begin_index_str)
                    break

        print('\n------------------------')
        print('begin_index_str:', begin_index_str)
        print('end_index_str:', end_index_str)
        print('------------------------\n')

        adjust_output_folders_names(args)
        print('args.i:', args.i)
        print('args.a:', args.a)
        print('args.o:', args.o)
        
        name_file_faces_not_det = 'files_faces_not_detected.txt'
        path_file_faces_not_det = os.path.join(os.path.dirname(args.o), name_file_faces_not_det)
        print('path_file_faces_not_det:', path_file_faces_not_det)

        sub_folders = sub_folders[begin_index_str:end_index_str]
        for s, sub_folder in enumerate(sub_folders):
            if not args.dont_detect_face:
                print('\nExtracting face crops...')
                paths = process_BERNARDO(sub_folder, args, app, args.ignore_face_not_det, path_file_faces_not_det)
            else:
                print('\nExtracting face crops (no face detection)...')
                paths = process_BERNARDO_no_face_det(sub_folder, args, app)
            print('')


            for p, path in enumerate(paths):
                start_time = time.time()
                print(f'divs: {args.div}')
                print(f'begin_parts: {begin_parts}')
                print(f'  end_parts: {end_parts}')
                print(f'part {args.part} ({begin_parts[args.part]}:{end_parts[args.part]}  size {end_parts[args.part]-begin_parts[args.part]}) - sample {p}/{len(paths)-1} - subfolder {s}/{len(sub_folders)}')
                print(f'Reconstructing \'{path}\'')

                images, arcface = to_batch(path)
                codedict = mica.encode(images, arcface)
                opdict = mica.decode(codedict)
                embedd_2D_arcface = codedict['arcface']
                meshes = opdict['pred_canonical_shape_vertices']
                code = opdict['pred_shape_code']
                lmk = mica.flame.compute_landmarks(meshes)

                mesh = meshes[0]
                landmark_51 = lmk[0, 17:]
                landmark_7 = landmark_51[[19, 22, 25, 28, 16, 31, 37]]
                rendering = mica.render.render_mesh(mesh[None])
                image = (rendering[0].cpu().numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
                image = np.minimum(np.maximum(image, 0), 255).astype(np.uint8)

                path_subj_output_folder = os.path.dirname(path.replace(args.a, args.o))

                Path(path_subj_output_folder).mkdir(exist_ok=True, parents=True)

                sample_file_name = path.split('/')[-1].split('.')[0]
                path_sample_output_folder = os.path.join(path_subj_output_folder, sample_file_name)

                Path(path_sample_output_folder).mkdir(exist_ok=True, parents=True)

                dst = path_sample_output_folder
                pointcloud = verts=mesh.cpu() * 1000.0   # save in millimeters
                lmk_7_cpu = landmark_7.cpu().numpy() * 1000.0
                lmk68_cpu = lmk.cpu().numpy() * 1000.0

                cv2.imwrite(f'{dst}/render.jpg', image)

                if not args.save_only_sampled:

                    np.save(f'{dst}/embedd_2D_arcface', embedd_2D_arcface.cpu().numpy())
                    save_ply(f'{dst}/mesh.ply', pointcloud, faces=faces)  
                    save_obj(f'{dst}/mesh.obj', pointcloud, faces=faces)
                    np.save(f'{dst}/identity', code[0].cpu().numpy())
                    np.save(f'{dst}/kpt7', lmk_7_cpu)
                    np.save(f'{dst}/kpt68', lmk68_cpu)

                else:
                    nosetip = lmk68_cpu[0,30,:]
                    radius = 100   # 10 cm
                    filtered_pc = filter_points_in_sphere(pointcloud, nosetip, radius)
                    centralized_pc = translate_point_cloud(filtered_pc, nosetip)
                    util.write_obj(f'{dst}/mesh_centralized_nosetip_croped_radius={radius}.obj', centralized_pc)
                    np.save(f'{dst}/mesh_centralized_nosetip_croped_radius={radius}.npy', centralized_pc)
                    util.write_obj(f'{dst}/kpt7.obj', lmk_7_cpu)
                    util.write_obj(f'{dst}/lmk68.obj', lmk68_cpu[0])
                    np.save(f'{dst}/embedd_2D_arcface', embedd_2D_arcface.cpu().numpy())
                    np.save(f'{dst}/identity', code[0].cpu().numpy())

                elapsed_time = time.time() - start_time
                print(f'Elapsed time: {elapsed_time} seconds')
                print('---------------')


            logger.info(f'Processing finished. Results has been saved in {args.o}')
            print('------------------------------')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MICA - Towards Metrical Reconstruction of Human Faces')
    parser.add_argument('-file_list', type=str, default='', help='')

    parser.add_argument('-i', default='/datasets2/frcsyn_wacv2024/datasets/synthetic/GANDiffFace_crops', type=str, help='Input folder with images')

    parser.add_argument('-o', default='/datasets2/frcsyn_wacv2024/datasets/3D_reconstruction_MICA/synthetic/GANDiffFace_crops', type=str, help='Output folder')
    
    parser.add_argument('-a', default='', type=str, help='Processed images for MICA input')
    
    parser.add_argument('-m', default='data/pretrained/mica.tar', type=str, help='Pretrained model path')

    parser.add_argument('-str_begin', default='', type=str, help='Substring to find and start processing')
    parser.add_argument('-str_end', default='', type=str, help='Substring to find and stop processing')
    parser.add_argument('-str_pattern', default='', type=str, help='Substring to find and stop processing')

    parser.add_argument('-div', default=1, type=int, help='How many parts to divide paths list (useful to paralelize process)')
    parser.add_argument('-part', default=0, type=int, help='Specific part to process (works only if -div > 1)')

    parser.add_argument('-dont_detect_face', action='store_true', help='')
    parser.add_argument('-ignore_face_not_det', action='store_true', help='')
    parser.add_argument('-save_lmk', action='store_true', help='')

    parser.add_argument('-save_only_sampled', action='store_true', help='')

    args = parser.parse_args()
    cfg = get_cfg_defaults()

    deterministic(42)
    main(cfg, args)
