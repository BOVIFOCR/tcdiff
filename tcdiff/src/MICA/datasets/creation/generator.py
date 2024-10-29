















import sys
import os
from glob import glob
from multiprocessing import Pool
from pathlib import Path
from typing import List

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import face_align
from loguru import logger
from tqdm import tqdm
import shutil

from instances.instance import Instance
from util import get_image, get_center, get_arcface_input


def _transfer(src, dst):
    src.parent.mkdir(parents=True, exist_ok=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.system(f'cp {str(src)} {str(dst)}')


def _copy(payload):
    instance, func, target, transform_path = payload
    files = func()
    for actor in files.keys():
        for file in files[actor]:
            _transfer(Path(file), Path(instance.get_dst(), target, actor, transform_path(file)))


class Generator:
    def __init__(self, instances):
        self.instances: List[Instance] = instances
        self.ARCFACE = '_arcface_input'

    def copy(self):
        tqdm.write('Start copying...')
        for instance in tqdm(self.instances):
            payloads = [(instance, instance.get_images, 'images', instance.transform_path)]
            
            tqdm.write('payloads: ' + str(payloads))
            tqdm.write('instance: ' + str(instance))
            
            with Pool(processes=len(payloads)) as pool:
                for _ in tqdm(pool.imap_unordered(_copy, payloads), total=len(payloads)):
                    pass

    def preprocess(self):
        logger.info('Start preprocessing...')
        for instance in tqdm(self.instances):
            instance.preprocess()

    def arcface(self):
        app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(224, 224))

        logger.info('Start arcface...')

        for instance in self.instances:
            
            src = instance.get_src()

            tqdm.write('src: ' + src)
            tqdm.write('path: ' + f'{src}*')

            images_without_faces = []

            for image_path in tqdm(sorted(glob(f'{src}/images/*/*'))):
                image_path = image_path.replace('//', '/')
                tqdm.write('image_path: ' + str(image_path))
                
                dst = image_path.replace('images', self.ARCFACE)
                assert src != dst
                
                tqdm.write('dst: ' + str(dst))
                
                Path(dst).parent.mkdir(exist_ok=True, parents=True)
                for img in instance.transform_image(get_image(image_path[0:-4])):
                    bboxes, kpss = app.det_model.detect(img, max_num=0, metric='default')

                    if bboxes.shape[0] == 0:
                        images_without_faces.append(image_path)
                        continue

                    i = get_center(bboxes, img)
                    bbox = bboxes[i, 0:4]
                    det_score = bboxes[i, 4]
                    if det_score < instance.get_min_det_score():
                        continue
                    kps = None
                    if kpss is not None:
                        kps = kpss[i]
                    face = Face(bbox=bbox, kps=kps, det_score=det_score)
                    blob, aimg = get_arcface_input(face, img)
                    np.save(dst[0:-4], blob)
                    cv2.imwrite(dst, face_align.norm_crop(img, landmark=face.kps, image_size=224))

                subj = image_path.split('/')[-2]
                dir_src_npz_file = '/'.join(image_path.split('/')[:-3]) + '/FLAME_parameters/' + subj
                npz_pattern_to_search = dir_src_npz_file + '/' + '*.npz'

                found_file = glob(npz_pattern_to_search, recursive=True)

                if len(found_file) == 0:
                    print('Error, file not found:', file_name_to_search)
                    sys.exit(0)
                elif len(found_file) > 1:
                    print('Error, multiple files found:', found_file)
                    sys.exit(0)

                found_file = found_file[0]
                tqdm.write('found_file: ' + str(found_file))
                output_npz_file = '/'.join(dst.split('/')[0:-1]) + '/' + found_file.split('/')[-1]
                tqdm.write('output_npz_file: ' + str(output_npz_file))
                assert found_file != output_npz_file
                shutil.copyfile(found_file, output_npz_file)

            if len(images_without_faces) > 0:
                tqdm.write('\n\nIMAGES WITHOUT FACES:')
                for path in images_without_faces:
                    tqdm.write('path: ' + str(path))



    def run(self):
        self.copy()
        self.preprocess()
        self.arcface()
