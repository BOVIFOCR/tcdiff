
















import sys
import numpy as np
from torch.utils.data import ConcatDataset
import torch

try:
    from datasets.base import BaseDataset
    from datasets.base_multitask_facerecognition import BaseDataset_MultitaskFaceRecognition
except ModuleNotFoundError:
    from MICA.datasets.base import BaseDataset
    from MICA.datasets.base_multitask_facerecognition import BaseDataset_MultitaskFaceRecognition


def get_imagelabel_from_imagename(imagename, labels_map):
    imagelabel = [None] * len(imagename)
    for i, name in enumerate(imagename):
        imagelabel[i] = labels_map[name]
    return np.array(imagelabel)

def get_onehotvector_from_imagelabel1(imagelabel, num_classes):
    one_hot_labels = np.eye(num_classes)[imagelabel]
    return one_hot_labels

def get_onehotvector_from_imagelabel2(imagelabel, num_classes):

    one_hot_labels = torch.nn.functional.one_hot(torch.from_numpy(imagelabel), num_classes)
    return one_hot_labels

def get_all_indexes_in_list(value, values_list):
    indexes = np.where(np.array(values_list) == value)[0]
    return indexes

def get_labels_map(train_dataset, val_dataset):
    i = 0
    all_actors_name = []
    labels_map = {}


    for j, dataset in enumerate(train_dataset.datasets):
        actors_name = list(dataset.face_dict.keys())
        all_actors_name += actors_name
        for name in actors_name:
            if not name in labels_map:
                labels_map[name] = i
                i += 1
            else:
                print('ERROR: repeated actor name:', name)
                sys.exit(0)

    return labels_map


def build_train_multitask_facerecognition(config, device, cfg):
    data_list = []
    total_images = 0
    for dataset in config.training_data:
        dataset_name = dataset.upper()
        config.n_train = np.Inf
        if type(dataset) is list:
            dataset_name, n_train = dataset
            config.n_train = n_train

        dataset = BaseDataset_MultitaskFaceRecognition(name=dataset_name, config=config, device=device, isEval=False, cfg=cfg)

        data_list.append(dataset)
        total_images += dataset.total_images

    return ConcatDataset(data_list), total_images


def build_val_multitask_facerecognition(config, device, cfg):
    data_list = []
    total_images = 0
    for dataset in config.eval_data:
        dataset_name = dataset.upper()
        config.n_train = np.Inf
        if type(dataset) is list:
            dataset_name, n_train = dataset
            config.n_train = n_train

        dataset = BaseDataset_MultitaskFaceRecognition(name=dataset_name, config=config, device=device, isEval=True, cfg=cfg)

        data_list.append(dataset)
        total_images += dataset.total_images

    return ConcatDataset(data_list), total_images




def build_train(config, device):
    data_list = []
    total_images = 0
    for dataset in config.training_data:
        dataset_name = dataset.upper()
        config.n_train = np.Inf
        if type(dataset) is list:
            dataset_name, n_train = dataset
            config.n_train = n_train

        dataset = BaseDataset(name=dataset_name, config=config, device=device, isEval=False)

        data_list.append(dataset)
        total_images += dataset.total_images

    return ConcatDataset(data_list), total_images


def build_val(config, device):
    data_list = []
    total_images = 0
    for dataset in config.eval_data:
        dataset_name = dataset.upper()
        config.n_train = np.Inf
        if type(dataset) is list:
            dataset_name, n_train = dataset
            config.n_train = n_train

        dataset = BaseDataset(name=dataset_name, config=config, device=device, isEval=True)
        data_list.append(dataset)
        total_images += dataset.total_images

    return ConcatDataset(data_list), total_images
