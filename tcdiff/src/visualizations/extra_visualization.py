import pandas as pd
import torch
import numpy as np
import os, sys
import cv2
from src.visualizations.sample_visual import render_condition
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time


class ListDatasetWithIndex(Dataset):
    def __init__(self, img_list, flip_color=True):
        super(ListDatasetWithIndex, self).__init__()

        self.img_list = img_list
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.flip_color = flip_color

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])
        if img is None:
            raise ValueError(self.img_list[idx])
        img = img[:, :, :3]
        if self.flip_color:
            img = img[:, :, ::-1]
        img = Image.fromarray(img)
        img = self.transform(img)
        # return img, idx
        return img, idx, self.img_list[idx]


class StyleIdDataset(Dataset):

    def __init__(self, labels_splits, names_splits,
                 style_index_splits, id_index_splits,
                 style_dataset, id_dataset):
        super(StyleIdDataset, self).__init__()

        self.labels_splits = labels_splits
        self.names_splits = names_splits
        self.style_index_splits = style_index_splits
        self.id_index_splits = id_index_splits
        self.style_dataset = style_dataset
        self.id_dataset = id_dataset

        assert len(self.labels_splits) == len(self.names_splits)
        assert len(self.labels_splits) == len(self.style_index_splits)
        assert len(self.labels_splits) == len(self.id_index_splits)


    def __len__(self):
        return len(self.labels_splits)

    '''
    def __getitem__(self, idx):
        labels = self.labels_splits[idx]
        names = self.names_splits[idx]

        id_indexes = self.id_index_splits[idx]
        id_images = torch.stack([self.id_dataset[idx.item()][0] for idx in id_indexes])

        style_indexes = self.style_index_splits[idx]
        style_images = []
        for idx in style_indexes:
            batch = self.style_dataset[idx.item()]
            if isinstance(batch, dict):
                img = batch['image']
            else:
                img = batch[0]
            style_images.append(img)
        style_images = torch.stack(style_images)


        return id_images, style_images, labels, names
    '''

    def __getitem__(self, idx):
        labels = self.labels_splits[idx]
        names = self.names_splits[idx]

        id_indexes = self.id_index_splits[idx]
        id_images_list = [self.id_dataset[idx.item()] for idx in id_indexes]
        id_images = torch.stack([id_image[0] for id_image in id_images_list])
        id_images_paths = [id_image[2] for id_image in id_images_list]

        style_indexes = self.style_index_splits[idx]
        style_images = []
        style_images_paths = []
        for idx in style_indexes:
            batch = self.style_dataset[idx.item()]
            if isinstance(batch, dict):
                img = batch['image']
            else:
                img = batch[0]
                img_path = batch[2]
            style_images.append(img)
            style_images_paths.append(img_path)
        style_images = torch.stack(style_images)
        
        return id_images, style_images, labels, names, id_images_paths, style_images_paths


def batched_label_name_list(batch_size, num_subject, num_image_per_subject, num_partition, partition_idx):
    labels = torch.tensor([[i] * num_image_per_subject for i in range(num_subject)]).view(-1)
    names = torch.arange(num_image_per_subject).repeat(num_subject)
    if num_partition > 1:
        labels = np.array_split(labels, num_partition)[partition_idx]
        names = np.array_split(names, num_partition)[partition_idx]
    labels_split = torch.split(labels, batch_size)
    names_split = torch.split(names, batch_size)
    return labels_split, names_split


def style_image_sampler(style_sampling_method, num_image_per_subject, id_index_splits, names_splits, id_dataset,
                        style_dataset, pl_module, idx_splits=None):
    if 'feature_sim' in style_sampling_method:
        sim_df_path = 'sub_projects/make_similarity_list/make_similarity_list/center_ir_101_adaface_webface4m_faces_webface_112x112.pth'
        sim_df_dict = torch.load(os.path.join(pl_module.hparams.paths.repo_root, sim_df_path))['similarity_df']
        id_dataloader = DataLoader(id_dataset, batch_size=64, num_workers=0, shuffle=False)
        topk_similar_centers_all = []
        center = pl_module.recognition_model.center.weight.data
        for id_batch in tqdm(id_dataloader, total=len(id_dataloader), desc='inferring id features'):
            id_features, spatial = pl_module.recognition_model(id_batch[0].to(pl_module.device))
            id_features = id_features / torch.norm(id_features, 2, -1, keepdim=True)
            id_center_cossim = id_features @ center.T
            topk_similar_centers = id_center_cossim.topk(num_image_per_subject, dim=1)[1]
            topk_similar_centers_all.append(topk_similar_centers.cpu())
        topk_similar_centers_all = torch.cat(topk_similar_centers_all, dim=0)

        style_index_splits = []
        for id_index_split, names_split in zip(id_index_splits, names_splits):
            batch_size = len(id_index_split)

            style_index_split = []
            for id_index, name in zip(id_index_split, names_split):
                center_candidates = topk_similar_centers_all[id_index]
                if style_sampling_method == 'feature_sim_center:topk_sampling_top1':
                    center_index = center_candidates[name % len(center_candidates)].item()
                    style_index = sim_df_dict[center_index].loc[0, 'data_index']
                elif style_sampling_method == 'feature_sim_center:top1_sampling_topk':
                    center_index = center_candidates[0].item()
                    sub_sim_df = sim_df_dict[center_index]
                    sub_sim_df = sub_sim_df[sub_sim_df['cossim'] > sub_sim_df['cossim'].quantile(0.1)]
                    style_index = sub_sim_df.sample(1)['data_index'].item()
                else:
                    raise ValueError('not correct style_sampling_method')

                style_index_split.append(style_index)
            style_index_split = torch.tensor(style_index_split)
            style_index_splits.append(style_index_split)

    elif style_sampling_method == 'list':
        style_index_splits = []
        for id_index_split, names_split in zip(id_index_splits, names_splits):
            batch_size = len(id_index_split)

            style_index_split = np.arange(batch_size)
            style_index_splits.append(style_index_split)
    elif style_sampling_method == 'random':
        style_index_splits = []
        for id_index_split, names_split in zip(id_index_splits, names_splits):
            batch_size = len(id_index_split)

            style_index_split = [np.random.randint(0, len(style_dataset), 1)[0] for i in range(batch_size)]
            style_index_splits.append(style_index_split)
    elif style_sampling_method == 'same_gender_same_race':
        casia_attr_pred_path = os.path.join(pl_module.hparams.paths.data_dir, 'datagen/casia_attributes/_predictions.csv')
        casia_attr_pred = pd.read_csv(casia_attr_pred_path, index_col=0)
        sample_attr_pred_path = os.path.join(pl_module.hparams.paths.data_dir, 'datagen/ddpm_attributes/_predictions.csv')
        sample_attr_pred = pd.read_csv(sample_attr_pred_path, index_col=0)
        sample_attr_pred['basename'] = sample_attr_pred['path'].apply(lambda x:os.path.basename(x))
        sample_attr_pred.set_index('basename', inplace=True)

        casia_attr_pred_groups = casia_attr_pred.groupby(['gender', 'race'])
        style_index_splits = []
        for id_index_split, names_split in tqdm(zip(id_index_splits, names_splits), total=len(names_splits), desc='sampling styles'):
            style_index_split = []
            for id in id_index_split:
                id_path = id_dataset.img_list[id.item()]
                attr_pred = sample_attr_pred.loc[os.path.basename(id_path)]
                candidate_style = casia_attr_pred_groups.get_group((attr_pred.gender, attr_pred.race))
                sampled_style = candidate_style.sample(1)
                style_index_split.append(sampled_style.idx.item())
            style_index_split = torch.tensor(style_index_split)
            style_index_splits.append(style_index_split)
    elif style_sampling_method == 'train_data':
        style_index_splits = idx_splits

    elif style_sampling_method == 'mapping':
        global_index = 0
        style_index_splits = []
        for id_index_split, names_split in zip(id_index_splits, names_splits):
            batch_size = len(id_index_split)

            # style_index_split = [np.random.randint(0, len(style_dataset), 1)[0] for i in range(batch_size)]
            style_index_split = []
            for i in range(batch_size):
                style_index_split.append(global_index)
                global_index += 1

            style_index_splits.append(torch.tensor(style_index_split))
        # print('style_index_splits:', style_index_splits)

    else:
        raise ValueError('not correct style sampling meth')

    return style_index_splits


def dataset_generate(pl_module, style_dataset, id_dataset, num_image_per_subject,
                     num_subject=10000, batch_size=64, num_workers=0, save_root='./', style_sampling_method='random',
                     num_partition=1, partition_idx=0, writer=None, start_label=-1, seed=440,
                     save_id_img=False, save_style_img=False):
    os.makedirs(save_root, exist_ok=True)
    print(save_root)


    labels_splits, names_splits = batched_label_name_list(batch_size, num_subject,
                                                          num_image_per_subject, num_partition, partition_idx)


    id_index_splits = labels_splits  # label name becomes index to sample id image from id_dataset


    style_index_splits = style_image_sampler(style_sampling_method, num_image_per_subject, id_index_splits,
                                             names_splits, id_dataset, style_dataset, pl_module)

    datagen_dataset = StyleIdDataset(labels_splits, names_splits,
                                     style_index_splits, id_index_splits,
                                     style_dataset, id_dataset)

    def collate_fn(data): return data[0]
    datagen_dataloader = DataLoader(datagen_dataset, num_workers=num_workers, batch_size=1, shuffle=False, collate_fn=collate_fn)

    for batch in tqdm(datagen_dataloader, total=len(datagen_dataloader), desc='Generating Dataset: '):
        start_time = time.time()
        # id_images, style_images, labels, names = batch
        id_images, style_images, labels, names, id_images_paths, style_images_paths = batch

        if torch.any(labels >= start_label):

            # plotting_images = sample_batch(id_images, style_images, pl_module, seed=labels[0].item())
            plotting_images, plotting_id_images, plotting_style_images = sample_batch_return_stylized_id_sty(id_images, style_images, pl_module, seed)

            end_time = time.time()
            total_elapsed_time = end_time - start_time
            print('    Total time: %.2fs    Time per sample: %.2fs' % (total_elapsed_time, total_elapsed_time/len(batch)))

            '''
            for image, label, name in zip(plotting_images, labels, names):
                save_name = f"{label.item()}/{name.item()}.jpg"
                if writer is not None:
                    writer.write(image, save_name)
                    writer.mark_done('image', save_name)
                else:
                    save_path = os.path.join(save_root, save_name)
                    print('    Saving output image:', save_path)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    cv2.imwrite(save_path, image)
            print('----------')
            '''

            for image, id_image, style_image, label, name, id_image_path, style_image_path in zip(plotting_images, plotting_id_images, plotting_style_images, labels, names, id_images_paths, style_images_paths):
                save_name = f"{label.item()}/{name.item()}.jpg"
                if writer is not None:
                    writer.write(image, save_name)
                    writer.mark_done('image', save_name)
                else:
                    print('    id_image_path:', id_image_path)
                    print('    style_image_path:', style_image_path)
                    
                    save_path = os.path.join(save_root, save_name)
                    print('    Saving output image:', save_path)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    cv2.imwrite(save_path, image)
                    
                    if save_id_img:
                        if name == 0:
                            id_image_path = os.path.join(save_root, save_name.replace('.jpg', '_id_image.jpg'))
                            print('    Saving id_image_path:', id_image_path)
                            cv2.imwrite(id_image_path, id_image)

                    if save_style_img:
                        style_image_path = os.path.join(save_root, save_name.replace('.jpg', '_style_image.jpg'))
                        print('    Saving style_image_path:', style_image_path)
                        cv2.imwrite(style_image_path, style_image)

                    print('    ---')
            print('---------------------')



def dataset_generate_mimic_train(pl_module, style_dataset, id_dataset, num_image_per_subject,
                     num_subject=10000, batch_size=64, num_workers=0, save_root='./', style_sampling_method='random',
                     num_partition=1, partition_idx=0, writer=None):
    os.makedirs(save_root, exist_ok=True)
    print(save_root)

    idx_splits = None
    print('style_sampling_method', style_sampling_method)

    if style_sampling_method in ['random', 'same_gender_same_race']:
        labels_splits, names_splits = batched_label_name_list(batch_size, num_subject,
                                                              num_image_per_subject, num_partition, partition_idx)

    elif style_sampling_method in ['train_data']:
        assert hasattr(style_dataset, 'record_info')
        casia_attr_pred_path = os.path.join(pl_module.hparams.paths.data_dir, 'datagen/casia_attributes/_predictions.csv')
        casia_attr_pred = pd.read_csv(casia_attr_pred_path, index_col=0)
        labels = torch.tensor(casia_attr_pred['target'])
        names = []
        count_dict = {}
        for label in labels:
            label = label.item()
            if label not in count_dict:
                count_dict[label] = []
            name = len(count_dict[label])
            names.append(name)
            count_dict[label].append(1)
        names = torch.tensor(names)
        idxes = torch.arange(len(names))
        if num_partition > 1:
            labels = np.array_split(labels, num_partition)[partition_idx]
            names = np.array_split(names, num_partition)[partition_idx]
            idxes = np.array_split(idxes, num_partition)[partition_idx]
        labels_splits = torch.split(labels, batch_size)
        names_splits = torch.split(names, batch_size)
        idx_splits = torch.split(idxes, batch_size)

        save_names = []
        for label_split, name_split, idx_split in tqdm(zip(labels_splits, names_splits, idx_splits), total=len(idx_splits)):
            for label, name, idx in zip(label_split, name_split, idx_split):
                save_name = f"{label.item()}/{name.item()}.jpg"
                save_names.append(save_name)
        assert len(np.unique(save_names)) == len(labels)

    else:
        raise ValueError()

    id_index_splits = labels_splits  # label name becomes index to sample id image from id_dataset


    style_index_splits = style_image_sampler(style_sampling_method, num_image_per_subject, id_index_splits,
                                             names_splits, id_dataset, style_dataset, pl_module, idx_splits)

    datagen_dataset = StyleIdDataset(labels_splits, names_splits,
                                     style_index_splits, id_index_splits,
                                     style_dataset, id_dataset)

    def collate_fn(data): return data[0]
    datagen_dataloader = DataLoader(datagen_dataset, num_workers=num_workers, batch_size=1, shuffle=False, collate_fn=collate_fn)

    for batch in tqdm(datagen_dataloader, total=len(datagen_dataloader), desc='Generating Dataset: '):
        id_images, style_images, labels, names = batch
        plotting_images = sample_batch(id_images, style_images, pl_module, seed=labels[0].item())
        for image, label, name in zip(plotting_images, labels, names):
            save_name = f"{label.item()}/{name.item()}.jpg"
            if writer is not None:
                writer.write(image, save_name)
                writer.mark_done('image', save_name)
            else:
                save_path = os.path.join(save_root, save_name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, image)



def sample_batch(id_images, style_images, pl_module, seed=None):

    batch = {'image': style_images,
             'class_label': torch.arange(len(style_images)),  # dummy
             'index': torch.arange(len(style_images)),  # dummy
             'orig': style_images,
             'id_image': id_images}

    if seed is not None:
        generator = torch.manual_seed(seed)
    else:
        generator = None
    pred_images = render_condition(batch, pl_module, sampler='ddim', between_zero_and_one=True,
                                   show_progress=False, generator=generator, mixing_batch=None,
                                   return_x0_intermediates=False)


    plotting_images = pred_images * 255
    return plotting_images[:, :, :, ::-1]



def sample_batch_return_stylized_id_sty(id_images, style_images, pl_module, seed=None):

    batch = {'image': style_images,
             'class_label': torch.arange(len(style_images)),  # dummy
             'index': torch.arange(len(style_images)),  # dummy
             'orig': style_images,
             'id_image': id_images}

    if seed is not None:
        generator = torch.manual_seed(seed)
    else:
        generator = None
    pred_images = render_condition(batch, pl_module, sampler='ddim', between_zero_and_one=True,
                                   show_progress=False, generator=generator, mixing_batch=None,
                                   return_x0_intermediates=False)

    # select which time to plot
    plotting_images = pred_images * 255
    plotting_id_images = (((id_images.permute(0, 2, 3, 1)+1)/2) * 255).numpy()
    plotting_style_images = (((style_images.permute(0, 2, 3, 1)+1)/2) * 255).numpy()
    return plotting_images[:, :, :, ::-1], plotting_id_images[:, :, :, ::-1], plotting_style_images[:, :, :, ::-1]
