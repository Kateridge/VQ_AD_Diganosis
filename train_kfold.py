import os
import numpy as np

from datasets.ADNI_HDF5 import DatasetTwo, get_data_two, ADNIALL_transform, ADNIALL_transform_GAN
from models.models import model_CLS_CNN, model_CLS_Transformer, model_CLS_CNN_Single, model_CLS_VQCNN_Single, \
    model_CLS_VQCNN_Transformer, model_CLS_VQCNN, model_CLS_VQCNN_MULTIMODEL, model_CLS_VQCNN_MULTIMODEL_Transformer, \
    model_CLS_CNN_MULTIMODEL_Transformer
from models.other_models import DeepGuidanceModel, pix2pixelGANModel, FGANModel, FGANModel_Single
from options import Option
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader

from utils import Logger


# get dataloaders according to splits
def setup_dataflow(train_idx, test_idx, isGAN=False):
    # further split training set and validation set
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=seed)

    train_data = [ADNI_data[i] for i in train_idx.tolist()]
    val_data = [ADNI_data[i] for i in val_idx.tolist()]
    test_data = [ADNI_data[i] for i in test_idx.tolist()]

    if opt.task == 'pMCIsMCI' and opt.add_ADCN:
        ADNI_ADCN_data = get_data_two(data_path=opt.dataroot, task='ADCN', include_missing=False, remove_CN=opt.remove_CN)
        train_data += ADNI_ADCN_data

    # create datasets
    if isGAN:
        train_dataset = DatasetTwo(data_dict=train_data, data_transform=train_transforms_GAN)
        val_dataset = DatasetTwo(data_dict=val_data, data_transform=val_transforms_GAN)
        test_dataset = DatasetTwo(data_dict=test_data, data_transform=val_transforms_GAN)
    else:
        train_dataset = DatasetTwo(data_dict=train_data, data_transform=train_transforms)
        val_dataset = DatasetTwo(data_dict=val_data, data_transform=val_transforms)
        test_dataset = DatasetTwo(data_dict=test_data, data_transform=val_transforms)
    print(f'Train Datasets: {len(train_dataset)}')
    print(f'Val Datasets: {len(val_dataset)}')
    print(f'Test Datasets: {len(test_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size)

    return train_loader, val_loader, test_loader


# initialize options and create output directory
opt = Option().parse()
save_dir = os.path.join('./checkpoints', opt.name)
logger_main = Logger(save_dir)

# load ADNI dataset
ADNI_data = get_data_two(data_path=opt.dataroot, task=opt.task, include_missing=False)
train_transforms, val_transforms = ADNIALL_transform(opt.aug)
train_transforms_GAN, val_transforms_GAN = ADNIALL_transform_GAN(False)

print('Successfully load datasets.....')

# prepare kfold splits
num_fold = 5
seed = 20221213
print(f'The random seed is {seed}')
kfold_splits = KFold(n_splits=num_fold, shuffle=True, random_state=seed)
if opt.model == 'pix2pixGAN' or opt.model == 'pix2pixGAN_Single':
    isGAN = True
else:
    isGAN = False
results = []

for fold_idx, (train_idx, test_idx) in enumerate(kfold_splits.split(np.arange(len(ADNI_data)))):
    logger_main.print_message(f'************Fold {fold_idx}************')
    train_dataloader, val_dataloader, test_dataloader = setup_dataflow(train_idx, test_idx, isGAN)
    if opt.model == 'CNN_Single':
        solver = model_CLS_CNN_Single(opt, fold_idx)
        # start training
        solver.start_train(train_dataloader, val_dataloader)
        res_fold = solver.start_test(test_dataloader)
    elif opt.model == 'CNN':
        solver = model_CLS_CNN(opt, fold_idx)
        res_fold = solver.start_train(train_dataloader, val_dataloader, test_dataloader)
    elif opt.model == 'Transformer':
        solver = model_CLS_Transformer(opt, fold_idx)
    elif opt.model == 'VQCNN_Single':
        solver = model_CLS_VQCNN_Single(opt, fold_idx)
        # start training
        solver.start_train(train_dataloader, val_dataloader)
        res_fold = solver.start_test(test_dataloader)
    elif opt.model == 'VQCNN':
        solver = model_CLS_VQCNN(opt, fold_idx)
    elif opt.model == 'VQTransformer':
        solver = model_CLS_VQCNN_Transformer(opt, fold_idx)
    elif opt.model == 'MULTIMODEL':
        solver = model_CLS_VQCNN_MULTIMODEL_Transformer(opt, fold_idx)
        # start training
        solver.start_train(train_dataloader, val_dataloader)
        res_fold = solver.start_test(test_dataloader, 3)
    elif opt.model == 'MULTIMODEL_withoutVQ':
        solver = model_CLS_CNN_MULTIMODEL_Transformer(opt, fold_idx)
        # start training
        solver.start_train(train_dataloader, val_dataloader)
        res_fold = solver.start_test(test_dataloader, 3)
    elif opt.model == 'DeepGuidance':
        solver = DeepGuidanceModel(opt, fold_idx)
        # start training
        solver.start_train(train_dataloader, val_dataloader)
        res_fold = solver.start_test(test_dataloader, 3)
    elif opt.model == 'pix2pixGAN':
        solver = pix2pixelGANModel(opt, fold_idx)
        solver.load_pretrained_GAN()
        # start training
        res_fold = solver.start_train(train_dataloader, val_dataloader, test_dataloader)
    elif opt.model == 'pix2pixGAN_Single':
        solver = pix2pixelGANModel(opt, fold_idx)
        solver.load_pretrained_GAN()
        # start training
        res_fold = solver.start_train(train_dataloader, val_dataloader, test_dataloader)
    elif opt.model == 'FGAN':
        solver = FGANModel(opt, fold_idx)
        solver.load_pretrained_GAN()
        # start training
        res_fold = solver.start_train(train_dataloader, val_dataloader, test_dataloader)
    elif opt.model == 'FGAN_Single':
        solver = FGANModel_Single(opt, fold_idx)
        solver.load_pretrained_GAN()
        # start training
        res_fold = solver.start_train(train_dataloader, val_dataloader, test_dataloader)
    else:
        solver = None

    # logging
    if opt.model == 'VQCNN_Single':
        logger_main.print_message(f'Test - Loss_CLS:{res_fold[0]:.4f} Loss_Q:{res_fold[1]:.4f} '
                                  f'ACC:{res_fold[2]:.4f} SEN:{res_fold[3]:.4f} '
                                  f'SPE:{res_fold[4]:.4f} F1:{res_fold[5]:.4f} '
                                  f'AUC:{res_fold[6]:.4f}')
    elif opt.model == 'VQTransformer' or opt.model == 'VQCNN':
        logger_main.print_message(f'Test - Loss_CLS:{res_fold[0]:.4f} '
                                  f'Loss_MRI_Q:{res_fold[1]:.4f} Loss_PET_Q:{res_fold[2]:.4f} '
                                  f'ACC:{res_fold[3]:.4f} SEN:{res_fold[4]:.4f} '
                                  f'SPE:{res_fold[5]:.4f} F1:{res_fold[6]:.4f} '
                                  f'AUC:{res_fold[7]:.4f}')
    else:
        logger_main.print_message(f'Test - Loss:{res_fold[0]:.4f} ACC:{res_fold[1]:.4f} '
                                  f'SEN:{res_fold[2]:.4f} SPE:{res_fold[3]:.4f} '
                                  f'F1:{res_fold[4]:.4f} AUC:{res_fold[5]:.4f}')
    results.append(res_fold)

results = np.array(results)
np.save(os.path.join(save_dir, 'results.npy'), results)
res_mean = np.mean(results, axis=0)
res_std = np.std(results, axis=0)
logger_main.print_message(f'************Final Results************')
if opt.model == 'VQCNN_Single':
    logger_main.print_message(f'loss_CLS: {res_mean[0]:.4f} +- {res_std[0]:.4f}\n'
                              f'loss_Q: {res_mean[1]:.4f} +- {res_std[1]:.4f}\n'
                              f'acc: {res_mean[2]:.4f} +- {res_std[2]:.4f}\n'
                              f'sen: {res_mean[3]:.4f} +- {res_std[3]:.4f}\n'
                              f'spe: {res_mean[4]:.4f} +- {res_std[4]:.4f}\n'
                              f'f1: {res_mean[5]:.4f} +- {res_std[5]:.4f}\n'
                              f'auc: {res_mean[6]:.4f} +- {res_std[6]:.4f}\n')
elif opt.model == 'VQTransformer' or opt.model == 'VQCNN':
    logger_main.print_message(f'loss_CLS: {res_mean[0]:.4f} +- {res_std[0]:.4f}\n'
                              f'loss_MRI_Q: {res_mean[1]:.4f} +- {res_std[1]:.4f}\n'
                              f'loss_PET_Q: {res_mean[2]:.4f} +- {res_std[2]:.4f}\n'
                              f'acc: {res_mean[3]:.4f} +- {res_std[3]:.4f}\n'
                              f'sen: {res_mean[4]:.4f} +- {res_std[4]:.4f}\n'
                              f'spe: {res_mean[5]:.4f} +- {res_std[5]:.4f}\n'
                              f'f1: {res_mean[6]:.4f} +- {res_std[6]:.4f}\n'
                              f'auc: {res_mean[7]:.4f} +- {res_std[7]:.4f}\n')
else:
    logger_main.print_message(f'loss: {res_mean[0]:.4f} +- {res_std[0]:.4f}\n'
                              f'acc: {res_mean[1]:.4f} +- {res_std[1]:.4f}\n'
                              f'sen: {res_mean[2]:.4f} +- {res_std[2]:.4f}\n'
                              f'spe: {res_mean[3]:.4f} +- {res_std[3]:.4f}\n'
                              f'f1: {res_mean[4]:.4f} +- {res_std[4]:.4f}\n'
                              f'auc: {res_mean[5]:.4f} +- {res_std[5]:.4f}\n')