import os, sys
import numpy as np
import glob
from argparse import ArgumentParser
import time
import pickle
import matplotlib.pyplot as plt
import cv2

import torch
from backbones import get_model

'''
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--cos_sim_path', type=str, default='/datasets2/bjgbiesseck/face_recognition/synthetic/stylegan2-ada_SFHQ/a_tiny_sample_130_images_selected_different_from_CASIA-WebFace/cosine_similarities')
    parser.add_argument('--cos_sim_ext', type=str, default='cosine_similarities.pkl')
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--num_parts_data', type=int, default=10)
    args = parser.parse_args()
    return args
'''

def cosine_similarity(embedd1, embedd2):
    embedd1[0] /= np.linalg.norm(embedd1[0])
    embedd2[0] /= np.linalg.norm(embedd2[0])
    sim = float(np.maximum(np.dot(embedd1[0],embedd2[0])/(np.linalg.norm(embedd1[0])*np.linalg.norm(embedd2[0])), 0.0))
    return sim


@torch.no_grad()
def get_face_embedd(model, img):
    embedd = model(img).numpy()
    return embedd


def load_trained_model(network, path_weights):
    net = get_model(network, fp16=False)
    net.load_state_dict(torch.load(path_weights))
    net.eval()
    return net


def load_normalize_img(img):
    img = cv2.imread(img)
    img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    return img


def save_heatmap_as_png(data, path, title, vmin=-1, vmax=1):
    """
    Save a 2D ndarray as a matplotlib heatmap PNG file.

    Parameters:
        data (ndarray): 2D array of data to be visualized.
        path (str): File path to save the PNG file.
        title (str): Title of the heatmap.
        vmin (float): Minimum value for colormap normalization.
        vmax (float): Maximum value for colormap normalization.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap='gray', vmin=vmin, vmax=vmax)


    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='black')


    plt.xticks(np.arange(data.shape[1]), np.arange(data.shape[1]))

    plt.yticks(np.arange(data.shape[0]), np.arange(data.shape[0]))

    plt.colorbar()
    plt.title(title)
    plt.savefig(path)
    plt.close()



def main(imgs_set1, imgs_set2):

    print(f'Loading trained model')
    model = load_trained_model('r100', './trained_models/ms1mv3_arcface_r100_fp16/backbone.pth')


    embedds_imgs_set1 = [None] * len(imgs_set1)
    cossims_intra_subj = np.zeros((len(imgs_set1),len(imgs_set1)), dtype=np.float32)
    for i in range(len(imgs_set1)):
        print(f'Computing embedds imgs_set1 - i: {i}', end='\r')
        img1 = load_normalize_img(imgs_set1[i])
        embedds_imgs_set1[i] = get_face_embedd(model, img1)
    print('')

    for i in range(len(embedds_imgs_set1)):
        embedd1 = embedds_imgs_set1[i]
        for j in range(len(embedds_imgs_set1)):
            embedd2 = embedds_imgs_set1[j]
            cossim = cosine_similarity(embedd1, embedd2)
            cossims_intra_subj[i,j] = cossim
            print(f'Intra-subject analisys - i: {i}  -  j: {j}  -  cossim: {cossim}', end='\r')
    print('')
    intra_subj1_path = 'intra_subj1.png'
    title1 = 'intra_subj1'
    print(f'Saving chart {intra_subj1_path}')
    save_heatmap_as_png(cossims_intra_subj, intra_subj1_path, title1)
    print('')





    embedds_imgs_set2 = [None] * len(imgs_set2)
    cossims_intra_subj = np.zeros((len(imgs_set2),len(imgs_set2)), dtype=np.float32)
    for i in range(len(imgs_set2)):
        print(f'Computing embedds imgs_set2 - i: {i}', end='\r')
        img1 = load_normalize_img(imgs_set2[i])
        embedds_imgs_set2[i] = get_face_embedd(model, img1)
    print('')

    for i in range(len(embedds_imgs_set2)):
        embedd1 = embedds_imgs_set2[i]
        for j in range(len(embedds_imgs_set2)):
            embedd2 = embedds_imgs_set2[j]
            cossim = cosine_similarity(embedd1, embedd2)
            cossims_intra_subj[i,j] = cossim
            print(f'Intra-subject analisys - i: {i}  -  j: {j}  -  cossim: {cossim}', end='\r')
    print('')
    intra_subj2_path = 'intra_subj2.png'
    title1 = 'intra_subj2'
    print(f'Saving chart {intra_subj2_path}')
    save_heatmap_as_png(cossims_intra_subj, intra_subj2_path, title1)
    print('')





    for i in range(len(embedds_imgs_set1)):
        embedd1 = embedds_imgs_set1[i]
        for j in range(len(embedds_imgs_set2)):
            embedd2 = embedds_imgs_set2[j]
            cossim = cosine_similarity(embedd1, embedd2)
            cossims_intra_subj[i,j] = cossim
            print(f'Inter subject analisys - i: {i}  -  j: {j}  -  cossim: {cossim}', end='\r')
    print('')
    inter_subj1_subj2_path = 'inter_subj1_subj2.png'
    title1_subj1_subj2 = 'inter_subj1_subj2'
    print(f'Saving chart {intra_subj1_path}')
    save_heatmap_as_png(cossims_intra_subj, inter_subj1_subj2_path, title1_subj1_subj2)
    
            
    sys.exit(0)

    

   


if __name__ == "__main__":


    imgs_set1 = [
        '/home/bjgbiesseck/GitHub/BOVIFOCR_tcdiff_synthetic_face/tcdiff/sample_images/id_images/sample_57.png',
        '/home/bjgbiesseck/GitHub/BOVIFOCR_tcdiff_synthetic_face/tcdiff/sample_images/style_images/woman/sample_5.png',
        '/home/bjgbiesseck/GitHub/BOVIFOCR_tcdiff_synthetic_face/tcdiff/sample_images/style_images/woman/sample_14.png',
        '/home/bjgbiesseck/GitHub/BOVIFOCR_tcdiff_synthetic_face/tcdiff/sample_images/style_images/woman/sample_20.png',
        '/home/bjgbiesseck/GitHub/BOVIFOCR_tcdiff_synthetic_face/tcdiff/sample_images/style_images/woman/sample_45.png',
        '/home/bjgbiesseck/GitHub/BOVIFOCR_tcdiff_synthetic_face/tcdiff/sample_images/style_images/woman/sample_63.png',
        '/home/bjgbiesseck/GitHub/BOVIFOCR_tcdiff_synthetic_face/tcdiff/sample_images/style_images/woman/sample_69.png',
        '/home/bjgbiesseck/GitHub/BOVIFOCR_tcdiff_synthetic_face/tcdiff/generated_images/tcdiff_5x5/id:sample_57/sty:list_woman/0/0.jpg',
        '/home/bjgbiesseck/GitHub/BOVIFOCR_tcdiff_synthetic_face/tcdiff/generated_images/tcdiff_5x5/id:sample_57/sty:list_woman/0/1.jpg',
        '/home/bjgbiesseck/GitHub/BOVIFOCR_tcdiff_synthetic_face/tcdiff/generated_images/tcdiff_5x5/id:sample_57/sty:list_woman/0/2.jpg',
        '/home/bjgbiesseck/GitHub/BOVIFOCR_tcdiff_synthetic_face/tcdiff/generated_images/tcdiff_5x5/id:sample_57/sty:list_woman/0/3.jpg',
        '/home/bjgbiesseck/GitHub/BOVIFOCR_tcdiff_synthetic_face/tcdiff/generated_images/tcdiff_5x5/id:sample_57/sty:list_woman/0/4.jpg',
        '/home/bjgbiesseck/GitHub/BOVIFOCR_tcdiff_synthetic_face/tcdiff/generated_images/tcdiff_5x5/id:sample_57/sty:list_woman/0/5.jpg'
    ]

    imgs_set2 = [
        '/datasets2/bjgbiesseck/face_recognition/synthetic/stylegan2-ada_SFHQ/3D_reconstruction_MICA/images/arcface/images/SFHQ_pt3_00000010.jpg',
        '/home/bjgbiesseck/GitHub/BOVIFOCR_tcdiff_synthetic_face/tcdiff/sample_images/style_images/woman/sample_5.png',
        '/home/bjgbiesseck/GitHub/BOVIFOCR_tcdiff_synthetic_face/tcdiff/sample_images/style_images/woman/sample_14.png',
        '/home/bjgbiesseck/GitHub/BOVIFOCR_tcdiff_synthetic_face/tcdiff/sample_images/style_images/woman/sample_20.png',
        '/home/bjgbiesseck/GitHub/BOVIFOCR_tcdiff_synthetic_face/tcdiff/sample_images/style_images/woman/sample_45.png',
        '/home/bjgbiesseck/GitHub/BOVIFOCR_tcdiff_synthetic_face/tcdiff/sample_images/style_images/woman/sample_63.png',
        '/home/bjgbiesseck/GitHub/BOVIFOCR_tcdiff_synthetic_face/tcdiff/sample_images/style_images/woman/sample_69.png',
        '/home/bjgbiesseck/GitHub/BOVIFOCR_tcdiff_synthetic_face/tcdiff/generated_images/tcdiff_5x5/id:SFHQ_pt3_00000010/sty:list_woman/0/0.jpg',
        '/home/bjgbiesseck/GitHub/BOVIFOCR_tcdiff_synthetic_face/tcdiff/generated_images/tcdiff_5x5/id:SFHQ_pt3_00000010/sty:list_woman/0/1.jpg',
        '/home/bjgbiesseck/GitHub/BOVIFOCR_tcdiff_synthetic_face/tcdiff/generated_images/tcdiff_5x5/id:SFHQ_pt3_00000010/sty:list_woman/0/2.jpg',
        '/home/bjgbiesseck/GitHub/BOVIFOCR_tcdiff_synthetic_face/tcdiff/generated_images/tcdiff_5x5/id:SFHQ_pt3_00000010/sty:list_woman/0/3.jpg',
        '/home/bjgbiesseck/GitHub/BOVIFOCR_tcdiff_synthetic_face/tcdiff/generated_images/tcdiff_5x5/id:SFHQ_pt3_00000010/sty:list_woman/0/4.jpg',
        '/home/bjgbiesseck/GitHub/BOVIFOCR_tcdiff_synthetic_face/tcdiff/generated_images/tcdiff_5x5/id:SFHQ_pt3_00000010/sty:list_woman/0/5.jpg'
    ]



    main(imgs_set1, imgs_set2)