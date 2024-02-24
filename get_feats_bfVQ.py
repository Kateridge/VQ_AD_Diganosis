import numpy as np
import torch
import torch.nn.functional as F
from datasets.ADNI_HDF5 import DatasetTwo, get_data_two, ADNIALL_transform
from models.models import model_CLS_CNN, model_CLS_VQCNN_MULTIMODEL_Transformer, model_CLS_CNN_Single
from models.other_models import DeepGuidanceModel
from options import Option
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from einops import rearrange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
opt = Option().parse()

#   1) prepare data
ADNI_data = get_data_two(data_path='D:\\Datasets\\ADNI_ALL\\ADNI.hdf5', task='ADCN', include_missing=False)
train_transforms, val_transforms = ADNIALL_transform(False)
kfold_splits = KFold(n_splits=5, shuffle=True, random_state=20221213)
folds = {}
feats_all = torch.zeros([0, 64], device=device)
for fold_idx, (train_idx, test_idx) in enumerate(kfold_splits.split(np.arange(len(ADNI_data)))):
    test_data = [ADNI_data[i] for i in test_idx.tolist()]
    test_dataset = DatasetTwo(data_dict=test_data, data_transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1)
    # 2) get features
    solver = model_CLS_CNN_Single(opt, fold_idx)
    solver._load_checkpoint()
    CNN = solver.nets.CNN
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if opt.modality == 'MRI':
                IMG, _, _, _, DX = data
                IMG = IMG.to(device)
            elif opt.modality == 'PET':
                _, IMG, _, _, DX = data
                IMG, DX = IMG.to(device), DX.to(device)

            feats = CNN(IMG) # b c h w d
            # Average Pool
            feats = rearrange(feats, 'b c h w d -> (b h w d) c')

            feats_all = torch.cat([feats_all, feats], dim=0)
    break

torch.save(feats_all.cpu(), f'CNN_{opt.modality}_feats.pt')
