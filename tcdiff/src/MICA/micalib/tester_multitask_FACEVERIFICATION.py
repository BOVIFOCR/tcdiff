
















import os, sys
from glob import glob

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import face_align
from loguru import logger
from pytorch3d.io import save_ply
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp
from tqdm import tqdm

from configs.config import cfg
from utils import util

from face_recognition.dataloaders.mlfw_verif_pairs_imgs import MLFW_Verif_Pairs_Images
from face_recognition.dataloaders.lfw_verif_pairs_imgs import LFW_Verif_Pairs_Images
from face_recognition.dataloaders.talfw_verif_pairs_imgs import TALFW_Verif_Pairs_Images

input_mean = 127.5
input_std = 127.5


MLFW_PICTURES = '/datasets1/bjgbiesseck/MLFW/aligned'
MLFW_VERIF_PAIRS_LIST = '/datasets1/bjgbiesseck/MLFW/pairs.txt'


LFW_PICTURES = '/datasets1/bjgbiesseck/lfw_from_bin'





TALFW_PICTURES = '/datasets1/bjgbiesseck/TALFW_cropped_aligned'
TALFW_VERIF_PAIRS_LIST = '/datasets1/bjgbiesseck/lfw/pairs.txt'           # use LFW pairs as TALFW doesn't provide verification protocol


class TesterMultitaskFacerverification(object):
    def __init__(self, nfc_model, config=None, device=None):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config

        self.device = device
        self.batch_size = self.cfg.dataset.batch_size
        self.K = self.cfg.dataset.K
        self.render_mesh = True
        self.embeddings_lyhm = {}


        self.nfc = nfc_model.to(self.device)
        self.nfc.testing = True

        logger.info(f'[INFO]            {torch.cuda.get_device_name(device)}')

    def load_checkpoint(self, model_path):
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.device}

        checkpoint = torch.load(model_path, map_location)

        if 'arcface' in checkpoint:
            self.nfc.arcface.load_state_dict(checkpoint['arcface'])
        if 'flameModel' in checkpoint:
            self.nfc.flameModel.load_state_dict(checkpoint['flameModel'])
        if 'faceClassifier' in checkpoint:
            self.nfc.faceClassifier.load_state_dict(checkpoint['faceClassifier'])

        logger.info(f"[TESTER] Resume from {model_path}")

    def load_model_dict(self, model_dict):
        dist.barrier()

        self.nfc.canonicalModel.load_state_dict(model_dict['canonicalModel'])
        self.nfc.arcface.load_state_dict(model_dict['arcface'])

    def process_image(self, img, app):

        
        if not app is None:
            bboxes, kpss = app.det_model.detect(img, max_num=0, metric='default')
            if bboxes.shape[0] != 1:
                logger.error('Face not detected!')
                return images
            i = 0
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            aimg = face_align.norm_crop(img, landmark=face.kps)
        else:
            aimg = img

        blob = cv2.dnn.blobFromImages([aimg], 1.0 / input_std, (112, 112), (input_mean, input_mean, input_mean), swapRB=False)

        image = torch.tensor(blob[0])[None]
        return image


    def get_name(self, best_model, id):
        if '_' in best_model:
            name = id if id is not None else best_model.split('_')[-1][0:-4]
        else:
            name = id if id is not None else best_model.split('/')[-1][0:-4]
        return name


    def test_mlfw(self, best_model, id=None):
        self.load_checkpoint(best_model)
        name = self.get_name(best_model, id)
        self.mlfw(name)


    def test_lfw(self, best_model, id=None):
        self.load_checkpoint(best_model)
        name = self.get_name(best_model, id)
        self.lfw(name)


    def save_mesh(self, file, vertices):
        scaled = vertices * 1000.0
        save_ply(file, scaled.cpu(), self.nfc.render.faces[0].cpu())






    def cache_to_cuda(self, cache):
        for key in cache.keys():
            i0, i1, l = cache[key]
            cache[key] = (i0.to(self.device), i1.to(self.device), l.to(self.device))
        return cache


    def create_mlfw_cache(self):
        cache_file_name = 'test_mlfw_cache.pt'
        cache = {}

        file_ext = '.jpg'
        all_pairs, pos_pair_label, neg_pair_label = MLFW_Verif_Pairs_Images().load_pairs_samples_protocol_from_file(MLFW_VERIF_PAIRS_LIST, MLFW_PICTURES, file_ext)

        arcface = []
        for i, pair in enumerate(all_pairs):
            path_img0, path_img1, label_pair = pair
            print('\x1b[2K', end='')
            print(f'tester_multitask_FACEVERIFICATION - create_mlfw_cache - {i}/{len(all_pairs)-1} ({path_img0}, {path_img1}, {label_pair})', end='\r')
            img0 = imread(path_img0)[:, :, :3]
            img1 = imread(path_img1)[:, :, :3]
            
            img0_preproc = self.process_image(img0.astype(np.float32), app=None)
            img1_preproc = self.process_image(img1.astype(np.float32), app=None)
            
            cache[i] = (img0_preproc, img1_preproc, torch.tensor(int(label_pair)))

        return self.cache_to_cuda(cache)


    def create_lfw_cache(self, output_folder='lfw_cropped_aligned'):
        cache_file_name = 'test_lfw_cache.pt'
        cache = {}

        file_ext = '.png'
        all_pairs, pos_pair_label, neg_pair_label = LFW_Verif_Pairs_Images().load_pairs_samples_protocol_from_bin_folder(LFW_PICTURES, num_folds=10, num_type_pair_per_fold=300, file_ext=file_ext)

        arcface = []
        for i, pair in enumerate(all_pairs):
            path_img0, path_img1, label_pair = pair
            print('\x1b[2K', end='')
            print(f'tester_multitask_FACEVERIFICATION - create_lfw_cache - {i}/{len(all_pairs)-1} ({path_img0}, {path_img1}, {label_pair})', end='\r')
            img0 = imread(path_img0)[:, :, :3]
            img1 = imread(path_img1)[:, :, :3]

            img0_preproc = self.process_image(img0.astype(np.float32), app=None)
            img1_preproc = self.process_image(img1.astype(np.float32), app=None)

            cache[i] = (img0_preproc, img1_preproc, torch.tensor(int(label_pair)))

        return self.cache_to_cuda(cache)
    

    def create_lfw_cache_classes(self, output_folder='lfw_cropped_aligned'):
        cache_file_name = 'test_lfw_cache.pt'
        cache = {}

        file_ext = '.png'
        img_paths, img_labels_names, img_labels_nums = LFW_Verif_Pairs_Images().load_class_samples_from_bin_folder('/datasets1/bjgbiesseck/lfw_cropped_aligned', file_ext=file_ext)

        arcface = []
        for i in range(len(img_paths[:100])):
            img_path, img_label_name, img_label_num = img_paths[i], img_labels_names[i], img_labels_nums[i]
            print('\x1b[2K', end='')
            print(f'tester_multitask_FACEVERIFICATION - create_lfw_cache_classes - {i}/{len(img_paths)-1} ({img_path}, {img_label_name}, {img_label_num})', end='\r')
            
            img0 = imread(img_path)[:, :, :3]
            img0_preproc = self.process_image(img0.astype(np.float32), app=None)
            
            cache[i] = (img0_preproc.to(self.device), torch.tensor(int(img_label_num)).to(self.device))

        return cache


    def create_talfw_cache(self, output_folder='TALFW_cropped_aligned'):
        cache_file_name = 'test_TALFW_cache.pt'
        cache = {}

        file_ext = '.png'
        all_pairs, pos_pair_label, neg_pair_label = TALFW_Verif_Pairs_Images().load_pairs_samples_protocol_from_file(TALFW_VERIF_PAIRS_LIST, TALFW_PICTURES, file_ext)

        arcface = []
        for i, pair in enumerate(all_pairs):
            path_img0, path_img1, label_pair = pair
            print('\x1b[2K', end='')
            print(f'tester_multitask_FACEVERIFICATION - create_talfw_cache - {i}/{len(all_pairs)-1} ({path_img0}, {path_img1}, {label_pair})', end='\r')
            img0 = imread(path_img0)[:, :, :3]
            img1 = imread(path_img1)[:, :, :3]

            img0_preproc = self.process_image(img0.astype(np.float32), app=None)
            img1_preproc = self.process_image(img1.astype(np.float32), app=None)

            cache[i] = (img0_preproc, img1_preproc, torch.tensor(int(label_pair)))

        return self.cache_to_cuda(cache)


    def get_face_embeddings(self, cache, flip_img=False):
        cache_keys = list(cache.keys())



        codedict = self.nfc.encode(images=None, arcface_imgs=cache[0][0])
        opdict = self.nfc.decode(codedict, 0)
        face_embedd = opdict['pred_shape_code']


        
        face_embedds = torch.zeros(len(cache_keys), face_embedd.size()[1])
        face_labels = torch.zeros(len(cache_keys))



        for i, key in enumerate(cache_keys):
            img0, label_num = cache[key]
            with torch.no_grad():

                if flip_img:
                    img0_flip = torch.flip(torch.permute(torch.clone(img0), (0, 2, 3, 1)), [2])







                    img0_flip = torch.permute(img0_flip, (0, 3, 1, 2))






                    codedict = self.nfc.encode(images=None, arcface_imgs=torch.cat([img0, img0_flip], dim=0))
                    opdict = self.nfc.decode(codedict, 0)
                    face_embedd = opdict['pred_shape_code']
                    face_embedd = F.normalize(face_embedd)


                    face_embedd[0] += face_embedd[1]


                    embedd0 = face_embedd[0]

                else:

                    codedict = self.nfc.encode(images=None, arcface_imgs=img0)
                    opdict = self.nfc.decode(codedict, 0)
                    face_embedd = opdict['pred_shape_code']

                    embedd0 = face_embedd[0]


                face_embedds[i] = embedd0
                face_labels[i] = label_num

                print('\x1b[2K', end='')
                print(f'tester_multitask_FACEVERIFICATION - get_face_embeddings - {i}/{len(cache_keys)-1} - label_num: {label_num}', end='\r')

        print()
        return face_embedds, face_labels




    def get_all_distances(self, cache, flip_img=False):
        cache_keys = list(cache.keys())
        cos_sims = torch.zeros(len(cache_keys))

        for i, key in enumerate(cache_keys):
            img0, img1, pair_label = cache[key]
            with torch.no_grad():

                if flip_img:
                    img0_flip = torch.flip(torch.permute(torch.clone(img0), (0, 2, 3, 1)), [2])
                    img1_flip = torch.flip(torch.permute(torch.clone(img1), (0, 2, 3, 1)), [2])






                    img0_flip = torch.permute(img0_flip, (0, 3, 1, 2))
                    img1_flip = torch.permute(img1_flip, (0, 3, 1, 2))




                    codedict = self.nfc.encode(images=None, arcface_imgs=torch.cat([img0, img0_flip, img1, img1_flip], dim=0))
                    opdict = self.nfc.decode(codedict, 0)
                    face_embedd = opdict['face_embedd']
                    face_embedd = F.normalize(face_embedd)


                    face_embedd[0] += face_embedd[1]
                    face_embedd[2] += face_embedd[3]
                    embedd0, embedd1 = face_embedd[0], face_embedd[2]

                else:
                    codedict = self.nfc.encode(images=None, arcface_imgs=torch.cat([img0, img1], dim=0))
                    opdict = self.nfc.decode(codedict, 0)
                    face_embedd = opdict['face_embedd']
                    embedd0, embedd1 = face_embedd[0], face_embedd[1]

                cos_sims[i] = torch.sum( torch.square( F.normalize(torch.unsqueeze(embedd0, 0)) - F.normalize(torch.unsqueeze(embedd1, 0)) ) )

                print('\x1b[2K', end='')
                print(f'tester_multitask_FACEVERIFICATION - get_all_distances - {i}/{len(cache_keys)-1} - pair_label: {pair_label}, cos_sims[{i}]: {cos_sims[i]}', end='\r')

        print(f'\ncos_sims.min():', cos_sims.min(), '   cos_sims.max():', cos_sims.max())
        return cos_sims
    


    def find_best_treshold(self, cache, cos_sims):
        best_tresh = 0
        best_acc = 0
        

        start, end, step = 0, 4, 0.01    # used in insightface code

        treshs = torch.arange(start, end+step, step)
        for i, tresh in enumerate(treshs):
            tresh = torch.round(tresh, decimals=3)
            tp, fp, tn, fn, acc = 0, 0, 0, 0, 0
            for j, cos_sim in enumerate(cos_sims):
                _, _, pair_label = cache[j]
                if pair_label == 1:
                    if cos_sim < tresh:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if cos_sim >= tresh:
                        tn += 1
                    else:
                        fp += 1

            acc = round((tp + tn) / (tp + tn + fp + fn), 4)


            if acc > best_acc:
                best_acc = acc
                best_tresh = tresh

            print('\x1b[2K', end='')
            print(f'tester_multitask_FACEVERIFICATION - {i}/{len(treshs)-1} - tresh: {tresh}', end='\r')

        return best_tresh, best_acc



    def save_similarities_to_file(self, path_file, cos_sims, cache):
        pair_labels = np.zeros((len(list(cache.keys())),), dtype=int)
        for i in range(len(list(cache.keys()))):
            pair_labels[i] = cache[i][2].cpu().numpy()
        data = {'cos-sims': cos_sims.numpy(),
                'pair_labels': pair_labels}
        np.save(path_file, data)
        



    def evaluate_model(self, checkpoint='', dataset_name='', args=None):
        logger.info(f"[TESTER] {dataset_name} testing has begun!")
        if checkpoint.endswith('.tar') and os.path.isfile(checkpoint):
            self.load_checkpoint(checkpoint)
            name = self.get_name(checkpoint, id)
        self.nfc.eval()

        logger.info(f"[TESTER] Creating {dataset_name} cache...")

        if dataset_name.upper() == 'MLFW':
            cache = self.create_mlfw_cache()
        elif dataset_name.upper() == 'LFW':
            cache = self.create_lfw_cache()
        elif dataset_name.upper() == 'TALFW':
            cache = self.create_talfw_cache()
        else:
            logger.error('[TESTER] Test dataset was not specified: ' + str(dataset_name))
            sys.exit(0)

        logger.info(f"Computing pair distances...")
        cos_sims = self.get_all_distances(cache)

        path_cos_sims = '/'.join(args.cfg.split('/')[:-2]) + '/output/' + '.'.join(args.cfg.split('/')[-1].split('.')[:-1])
        cos_sims_file = path_cos_sims + '/' + 'cos-sims_checkpoint=' + checkpoint.split('/')[-1] + '_dataset=' + dataset_name + '.npy'




        if not os.path.isdir(path_cos_sims):
            os.makedirs(path_cos_sims)
        logger.info(f"Saving cosine similarities to file: {cos_sims_file}")
        self.save_similarities_to_file(cos_sims_file, cos_sims.cpu(), cache)
        
        print()
        logger.info(f"Findind best treshold...")
        best_tresh, best_acc = self.find_best_treshold(cache, cos_sims)
        print(f'\nbest_tresh: {best_tresh},   best_acc: {best_acc}')



    def plot_reduced_face_embeddings(self, X, Y, path_file, save=True):
        import matplotlib.pyplot as plt




        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        im = ax.scatter(X[:,0], X[:,1], c=Y, cmap='Set3', alpha=1.0)

        ax.set_title(f'Reduced face embeddings - t-SNE')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')


        fig.tight_layout()

        fig.savefig(path_file)   # save the figure to file
        plt.close(fig)           # close the figure window
        




    def plot_face_embeddings(self, checkpoint='', dataset_name='', args=None):

        from sklearn.manifold import TSNE

        path_embeddings = '/'.join(args.cfg.split('/')[:-2]) + '/output/' + '.'.join(args.cfg.split('/')[-1].split('.')[:-1])
        embeddings_file = path_embeddings + '/' + 't-SNE_2D_checkpoint=' + checkpoint.split('/')[-1] + '_dataset=' + dataset_name + '.npy'
        
        if not os.path.exists(embeddings_file):
            logger.info(f"[TESTER] {dataset_name} plotting has begun!")
            if checkpoint.endswith('.tar') and os.path.isfile(checkpoint):
                self.load_checkpoint(checkpoint)
                name = self.get_name(checkpoint, id)
            self.nfc.eval()

            logger.info(f"[TESTER] Creating {dataset_name} cache...")

            
            if dataset_name.upper() == 'LFW':
                cache = self.create_lfw_cache_classes()




            else:
                logger.error('[TESTER] Test dataset was not specified: ' + str(dataset_name))
                sys.exit(0)

            logger.info(f"Computing face embeddings...")
            face_embedds, face_labels = self.get_face_embeddings(cache, flip_img=False)
            face_embedds = face_embedds.numpy().astype(float)
            face_labels = face_labels.numpy()







            logger.info(f"t-SNE: reducing dimensionality...")
            X_embedded = TSNE().fit_transform(face_embedds)

            print('X_embedded:', X_embedded.shape)

            if not os.path.isdir(path_embeddings):
                os.makedirs(path_embeddings)
            logger.info(f"Saving face embeddings to file: {embeddings_file}")
            data = {'face_embedds': face_embedds, 'face_embedds_reduced': X_embedded, 'face_labels': face_labels}
            np.save(embeddings_file, data)
        
        else:

            data = np.load(embeddings_file, allow_pickle=True).item()
            face_embedds = data['face_embedds']
            face_labels = data['face_labels']
            X_embedded = data['face_embedds_reduced']
        
        plot_embeddings_file = path_embeddings + '/' + 't-SNE_2D_checkpoint=' + checkpoint.split('/')[-1] + '_dataset=' + dataset_name + '.png'
        logger.info(f"Plotting reduced face embeddings: {plot_embeddings_file}")
        self.plot_reduced_face_embeddings(X_embedded, face_labels, plot_embeddings_file, save=True)


