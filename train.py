import os
from monai.data import partition_dataset
from datasets.ADNI_HDF5 import DatasetTwo, get_data_two, ADNIALL_transform
from models.models import model_CLS_CNN, model_CLS_Transformer, model_CLS_CNN_Single, model_CLS_VQCNN_Single, \
    model_CLS_VQCNN, model_CLS_VQCNN_Transformer, model_CLS_VQCNN_MULTIMODEL, model_CLS_VQCNN_MULTIMODEL_Transformer
from options import Option
from torch.utils.data import DataLoader
from utils import Logger


# get dataloaders according to splits
def setup_dataflow():
    # further split training set and validation set
    ADNI_partitions = partition_dataset(data=ADNI_data, ratios=[0.6, 0.2, 0.2], shuffle=True, seed=seed)

    train_data, val_data, test_data = ADNI_partitions[0], ADNI_partitions[1], ADNI_partitions[2]

    if opt.task == 'pMCIsMCI':
        ADNI_ADCN_data = get_data_two(data_path=opt.dataroot, task='ADCN', include_missing=False)
        train_data += ADNI_ADCN_data

    # create datasets
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

print('Successfully load datasets.....')

seed = 20221213
print(f'The random seed is {seed}')

train_dataloader, val_dataloader, test_dataloader = setup_dataflow()

if opt.model == 'CNN_Single':
    solver = model_CLS_CNN_Single(opt, 0)
elif opt.model == 'CNN':
    solver = model_CLS_CNN(opt, 0)
elif opt.model == 'Transformer':
    solver = model_CLS_Transformer(opt, 0)
elif opt.model == 'VQCNN_Single':
    solver = model_CLS_VQCNN_Single(opt, 0)
elif opt.model == 'VQCNN':
    solver = model_CLS_VQCNN(opt, 0)
elif opt.model == 'VQTransformer':
    solver = model_CLS_VQCNN_Transformer(opt, 0)
elif opt.model == 'MULTIMODEL':
    solver = model_CLS_VQCNN_MULTIMODEL(opt, 0)
else:
    solver = None

res_fold = solver.start_train(train_dataloader, val_dataloader, test_dataloader)

