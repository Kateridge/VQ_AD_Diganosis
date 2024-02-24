import numpy as np
from torch.utils.data import Dataset
import h5py
from monai.transforms import (Compose, EnsureType, RandRotate, RandZoom, RandFlip, EnsureChannelFirst, AddChannel,
                              SpatialPad)


def get_data_all(data_path, task, include_all=True):
    data = []

    if task == 'ADCN':
        target_label = {'AD': 1, 'CN': 0}
    elif task == 'pMCIsMCI':
        target_label = {'pMCI': 1, 'sMCI': 0}
    else:
        target_label = {'AD': 2, 'pMCI': 1, 'sMCI': 1, 'CN':0}

    with h5py.File(data_path, "r") as hf:
        for image_uid, g in hf.items():
            # get sub label
            DX = g.attrs['DX']
            # skip subjects
            if DX not in target_label.keys():
                continue
            sub_missing = g.attrs['missing'][:]
            # if not including missing, will skip subjects with missing modalities
            if (not include_all) and (not (sub_missing[0] and sub_missing[1] and sub_missing[2] and sub_missing[3])):
                continue
            # get sub image data and tabular data
            if sub_missing[0]:
                MRI = g['MRI'][:]
            else:
                MRI = 'missing'
            if sub_missing[1]:
                FDG = g['FDG'][:]
            else:
                FDG = 'missing'
            if sub_missing[2]:
                Amyloid = g['Amyloid'][:]
            else:
                Amyloid = 'missing'
            if sub_missing[3]:
                Tau = g['Tau'][:]
            else:
                Tau = 'missing'
            # get tabular data
            tabular = g['tabular'][:]
            # append data
            data.append(tuple([MRI, FDG, Amyloid, Tau, tabular, sub_missing, target_label[DX]]))
    return data


class DatasetALL(Dataset):
    def __init__(self, data_dict, data_transform):
        self.img_transform = data_transform
        tabular_all = []
        self.data = data_dict
        for _, _, _, _, tabular, _, _ in self.data:
            tabular_all.append(tabular)
        # calculate mean and std for tabular data
        tabular_all = np.array(tabular_all)
        self.meta = {'mean': np.nanmean(tabular_all, axis=0), 'std': np.nanstd(tabular_all, axis=0)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        MRI, FDG, Amyloid, Tau, tabular, sub_missing, DX = self.data[index]
        if type(FDG) is not np.ndarray:
            FDG = np.ones_like(MRI)
        if type(Amyloid) is not np.ndarray:
            Amyloid = np.ones_like(MRI)
        if type(Tau) is not np.ndarray:
            Tau = np.ones_like(MRI)
        # transform image data
        MRI, FDG, Amyloid, Tau = self.img_transform(MRI), self.img_transform(FDG), \
                                 self.img_transform(Amyloid), self.img_transform(Tau)
        # transform tabular data
        for i in range(tabular.shape[0]):
            if np.isnan(tabular[i]):
                tabular[i] = 0
            else:
                tabular[i] = (tabular[i] - self.meta['mean'][i]) / self.meta['std'][i]
        data_point = [MRI, FDG, Amyloid, Tau, tabular, sub_missing, DX]
        return tuple(data_point)


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
        train_transform = Compose([AddChannel(),
                                   RandFlip(prob=0.3, spatial_axis=0),
                                   RandRotate(prob=0.3, range_x=0.05),
                                   RandZoom(prob=0.3, min_zoom=0.95, max_zoom=1),
                                   EnsureType()])
    else:
        train_transform = Compose([AddChannel(), EnsureType()])
    test_transform = Compose([AddChannel(), EnsureType()])
    return train_transform, test_transform

def ADNIALL_transform_GAN(aug):
    if aug:
        train_transform = Compose([AddChannel(),
                                   RandFlip(prob=0.3, spatial_axis=0),
                                   RandRotate(prob=0.3, range_x=0.05),
                                   RandZoom(prob=0.3, min_zoom=0.95, max_zoom=1),
                                   EnsureType()])
    else:
        train_transform = Compose([AddChannel(), SpatialPad([128, 144, 128]), EnsureType()])
    test_transform = Compose([AddChannel(), SpatialPad([128, 144, 128]), EnsureType()])
    return train_transform, test_transform

# test_transform = Compose([
#             ScaleIntensity(),
#             AddChannel(),
#             EnsureType()
#         ])
# ADNI_data = get_data('/home/kateridge/Projects/Projects/datasets/ADNI/ADNI_3class.csv', 'ADCN')
# ADNI_dataset = DatasetHeterogeneous(ADNI_data, test_transform)
# print(ADNI_dataset.__getitem__(0))
# a = DatasetHeterogeneous('D:\\datasets\\ADNI_ALL\\ADNI.hdf5', test_transform, 'ADCN')
# print(len(a))
