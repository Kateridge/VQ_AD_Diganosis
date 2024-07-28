import numpy as np
from torch.utils.data import Dataset
import h5py
from monai.transforms import (Compose, EnsureType, RandRotate, RandZoom, RandFlip, EnsureChannelFirst,
                              SpatialPad)


def get_data_two(data_path, task, include_missing=True, remove_CN=False):
    data = []

    if task == 'ADCN':
        target_label = {'AD': 1, 'CN': 0}
    elif task == 'pMCIsMCI':
        target_label = {'pMCI': 1, 'sMCI': 0}
    else:
        target_label = {'AD': 2, 'pMCI': 1, 'sMCI': 0}

    with h5py.File(data_path, "r") as hf:
        for image_uid, g in hf.items():
            # get sub label
            DX = g.attrs['DX']
            # skip subjects
            if DX not in target_label.keys():
                continue
            if remove_CN and DX == 'CN':
                continue
            sub_missing = g.attrs['missing'][:]
            if not include_missing:
                # if not including missing, will skip subjects with missing modalities
                if not (sub_missing[0] and sub_missing[1]):
                    continue
                # get sub image data and tabular data
                MRI = g['MRI'][:]
                FDG = g['FDG'][:]
            else:
                # if including missing, will skip subjects with complete modalities
                if (sub_missing[0] and sub_missing[1]):
                    continue
                MRI = g['MRI'][:]
                FDG = 'missing'
            # get tabular data
            tabular = g['tabular'][:]
            # append data
            data.append(tuple([MRI, FDG, tabular, sub_missing, target_label[DX]]))
    return data


class DatasetTwo(Dataset):
    def __init__(self, data_dict, data_transform):
        self.img_transform = data_transform
        tabular_all = []
        self.data = data_dict
        for _, _, tabular, _, _ in self.data:
            tabular_all.append(tabular)
        # calculate mean and std for tabular data
        tabular_all = np.array(tabular_all)
        self.meta = {'mean': np.nanmean(tabular_all, axis=0), 'std': np.nanstd(tabular_all, axis=0)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        MRI, FDG, tabular, sub_missing, DX = self.data[index]
        if type(FDG) is not np.ndarray:
            FDG = np.ones_like(MRI)
        # transform image data
        MRI, FDG = self.img_transform(MRI), self.img_transform(FDG)
        # transform tabular data
        for i in range(tabular.shape[0]):
            if np.isnan(tabular[i]):
                tabular[i] = 0
            else:
                tabular[i] = (tabular[i] - self.meta['mean'][i]) / self.meta['std'][i]
        data_point = [MRI, FDG, tabular, sub_missing, DX]
        return tuple(data_point)


def get_data_single(data_path, task, modality='MRI'):
    target_label = []
    data = []
    M_list = {'MRI': 0, 'FDG': 1, 'Amyloid': 2, 'Tau': 3}

    if task == 'ADCN':
        target_label = {'AD': 1, 'CN': 0}
    elif task == 'pMCIsMCI':
        target_label = {'pMCI': 1, 'sMCI': 0}

    with h5py.File(data_path, "r") as hf:
        for image_uid, g in hf.items():
            # get sub label
            DX = g.attrs['DX']
            # skip subjects
            if DX not in target_label.keys():
                continue

            sub_missing = g.attrs['missing'][:]
            sub_missing = sub_missing[0:3]

            # get sub image data and tabular data
            if not sub_missing[M_list[modality]]:
                continue

            img = g[modality][:]
            tabular = g['tabular'][:]
            data.append(tuple([img, tabular, sub_missing, target_label[DX]]))
    return data


class DatasetSingle(Dataset):
    def __init__(self, data_dict, data_transform):
        self.img_transform = data_transform
        tabular_all = []
        self.data = data_dict
        for _, tabular, _, _ in self.data:
            tabular_all.append(tabular)
        # calculate mean and std for tabular data
        tabular_all = np.array(tabular_all)
        self.meta = {'mean': np.nanmean(tabular_all, axis=0), 'std': np.nanstd(tabular_all, axis=0)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        MRI, tabular, sub_missing, DX = self.data[index]
        MRI = self.img_transform(MRI)
        data_point = [MRI, tabular, sub_missing, DX]
        return tuple(data_point)


def ADNIALL_transform(aug):
    if aug:
        train_transform = Compose([EnsureChannelFirst(),
                                   RandFlip(prob=0.3, spatial_axis=0),
                                   RandRotate(prob=0.3, range_x=0.05),
                                   RandZoom(prob=0.3, min_zoom=0.95, max_zoom=1),
                                   EnsureType()])
    else:
        train_transform = Compose([EnsureChannelFirst(), EnsureType()])
    test_transform = Compose([EnsureChannelFirst(), EnsureType()])
    return train_transform, test_transform

def ADNIALL_transform_GAN(aug):
    if aug:
        train_transform = Compose([EnsureChannelFirst(),
                                   RandFlip(prob=0.3, spatial_axis=0),
                                   RandRotate(prob=0.3, range_x=0.05),
                                   RandZoom(prob=0.3, min_zoom=0.95, max_zoom=1),
                                   EnsureType()])
    else:
        train_transform = Compose([EnsureChannelFirst(), SpatialPad([128, 144, 128]), EnsureType()])
    test_transform = Compose([EnsureChannelFirst(), SpatialPad([128, 144, 128]), EnsureType()])
    return train_transform, test_transform
