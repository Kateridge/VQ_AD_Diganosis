import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import datetime

from einops import rearrange
from einops.layers.torch import Rearrange
from monai.data import decollate_batch
from monai.metrics import CumulativeAverage, ConfusionMatrixMetric, ROCAUCMetric
from monai.transforms import Compose, Activations, AsDiscrete

from munch import Munch
from sklearn.manifold import TSNE
from torch.nn import AdaptiveAvgPool1d, AdaptiveMaxPool1d

from models.networks import build_models
import utils

import matplotlib.pyplot as plt


def freeze_model(model):
    for (name, param) in model.named_parameters():
        param.requires_grad = False

def unfreeze_model(model):
    for (name, param) in model.named_parameters():
        param.requires_grad = True

class model_CLS_CNN_Single(nn.Module):
    def __init__(self, args, fold):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = build_models(args, 'CNN_Single')
        # create checkpoint dir
        self.fold = fold
        self.checkpoint_dir = os.path.join(args.checkpoints_dir, args.name, str(fold))
        if not os.path.exists(self.checkpoint_dir):
            utils.mkdir(self.checkpoint_dir)
        self.logger = utils.Logger(self.checkpoint_dir)
        # print network status
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        # create optimizers and checkpoints
        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                self.optims[net] = torch.optim.AdamW(
                    params=self.nets[net].parameters(), lr=args.lr, betas=(0.9, 0.999),
                    eps=1e-08, weight_decay=args.weight_decay)
            self.ckptios = [
                utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]
        else:
            self.ckptios = [
                utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]
        # to CUDA device
        self.to(self.device)
        # Initialize network parameters
        for name, network in self.named_children():
            self.logger.print_message('Initializing %s...' % name)
            network.apply(utils.he_init)
        # transform utils
        self.post_pred = Compose([Activations(softmax=True)])
        self.post_pred_argmax = Compose([AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([AsDiscrete(to_onehot=2)])
        # define evaluation metrics
        self.loss_epoch = CumulativeAverage()
        self.cls_metrics = ConfusionMatrixMetric(metric_name=["accuracy", 'sensitivity', 'specificity', 'f1 score'],
                                            include_background=False, reduction='mean')
        self.AUC = ROCAUCMetric(average='micro')

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def _save_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.save()

    def _load_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.load()

    def compute_CLS_loss(self, IMG, label):
        # forward CNN
        feats = self.nets.CNN(IMG)
        # Average Pool
        feats = F.adaptive_max_pool3d(feats, 1).view(IMG.shape[0], -1)
        # forward MLP
        logits = self.nets.CLS(feats)
        loss_CLS = F.cross_entropy(logits, label)
        return logits, loss_CLS

    def start_train(self, train_loaders, val_loader):
        args = self.args
        # use acc & f1 score to select model
        best_acc = 0.0
        best_f1 = 0.0
        best_epoch = 1
        self.logger.print_message('Start training...')
        start_time = time.time()

        for epoch in range(0, args.epochs):
            # training iterations
            for name, model in self.nets.items():
                model.train()
            for i, data in enumerate(train_loaders):
                # fetch images and labels
                if args.modality == 'MRI':
                    IMG, _, _, _, DX = data
                    IMG, DX = IMG.to(self.device), DX.to(self.device)
                elif args.modality == 'PET':
                    _, IMG, _, _, DX = data
                    IMG, DX = IMG.to(self.device), DX.to(self.device)
                else:
                    IMG, DX = None, None
                logits, loss_CLS = self.compute_CLS_loss(IMG, DX)
                # backward
                self._reset_grad()
                loss_CLS.backward()
                self.optims.CNN.step()
                self.optims.CLS.step()
                # compute training metrics
                #    epoch average loss
                self.loss_epoch.append(loss_CLS)
                #    cls metrics
                DX_onehot = [self.post_label(i) for i in decollate_batch(DX, detach=False)]
                y_pred_act = [self.post_pred(i) for i in decollate_batch(logits)]
                y_pred_act_onehot = [self.post_pred_argmax(i) for i in decollate_batch(logits)]
                self.cls_metrics(y_pred=y_pred_act_onehot, y=DX_onehot)
                self.AUC(y_pred=y_pred_act, y=DX_onehot)
            # log training metrics
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
            self.logger.print_message(f'Epoch {epoch + 1}/{args.epochs} - Elapsed time {elapsed}')
            loss_results = self.loss_epoch.aggregate()
            cm_result = self.cls_metrics.aggregate()
            AUC_results = self.AUC.aggregate()
            self.logger.print_message(f'Trainng    - Loss:{float(loss_results):.4f} '
                                      f'ACC:{float(cm_result[0]):.4f} '
                                      f'SEN:{float(cm_result[1]):.4f} '
                                      f'SPE:{float(cm_result[2]):.4f} '
                                      f'F1:{float(cm_result[3]):.4f} '
                                      f'AUC:{AUC_results:.4f}')
            # reset metrics
            self.loss_epoch.reset()
            self.cls_metrics.reset()
            self.AUC.reset()

            # validation iterations
            for name, model in self.nets.items():
                model.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    # fetch images and labels
                    if args.modality == 'MRI':
                        IMG, _, _, _, DX = data
                        IMG, DX = IMG.to(self.device), DX.to(self.device)
                    elif args.modality == 'PET':
                        _, IMG, _, _, DX = data
                        IMG, DX = IMG.to(self.device), DX.to(self.device)
                    else:
                        IMG, DX = None, None
                    logits, loss_CLS = self.compute_CLS_loss(IMG, DX)
                    # compute training metrics
                    #    epoch average loss
                    self.loss_epoch.append(loss_CLS)
                    #    cls metrics
                    DX_onehot = [self.post_label(i) for i in decollate_batch(DX, detach=False)]
                    y_pred_act = [self.post_pred(i) for i in decollate_batch(logits)]
                    y_pred_act_onehot = [self.post_pred_argmax(i) for i in decollate_batch(logits)]
                    self.cls_metrics(y_pred=y_pred_act_onehot, y=DX_onehot)
                    self.AUC(y_pred=y_pred_act, y=DX_onehot)
            # log validation metrics
            loss_results = self.loss_epoch.aggregate()
            cm_result = self.cls_metrics.aggregate()
            AUC_results = self.AUC.aggregate()
            self.logger.print_message(f'Validation - Loss:{float(loss_results):.4f} '
                                      f'ACC:{float(cm_result[0]):.4f} '
                                      f'SEN:{float(cm_result[1]):.4f} '
                                      f'SPE:{float(cm_result[2]):.4f} '
                                      f'F1:{float(cm_result[3]):.4f} '
                                      f'AUC:{AUC_results:.4f}')
            # save best model according to the validation results
            acc = float(cm_result[0].cpu())
            f1 = float(cm_result[3].cpu())
            if acc >= best_acc and f1 >= best_f1:
                self._save_checkpoint()
                best_acc = acc
                best_f1 = f1
                best_epoch = epoch + 1
            # reset metrics
            self.loss_epoch.reset()
            self.cls_metrics.reset()
            self.AUC.reset()

    def start_test(self, test_loader):
        args = self.args
        # start test iterations
        self._load_checkpoint()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # fetch images and labels
                if args.modality == 'MRI':
                    IMG, _, _, _, DX = data
                    IMG, DX = IMG.to(self.device), DX.to(self.device)
                elif args.modality == 'PET':
                    _, IMG, _, _, DX = data
                    IMG, DX = IMG.to(self.device), DX.to(self.device)
                else:
                    IMG, DX = None, None
                logits, loss_CLS = self.compute_CLS_loss(IMG, DX)
                # compute training metrics
                #    epoch average loss
                self.loss_epoch.append(loss_CLS)
                #    cls metrics
                DX_onehot = [self.post_label(i) for i in decollate_batch(DX, detach=False)]
                y_pred_act = [self.post_pred(i) for i in decollate_batch(logits)]
                y_pred_act_onehot = [self.post_pred_argmax(i) for i in decollate_batch(logits)]
                self.cls_metrics(y_pred=y_pred_act_onehot, y=DX_onehot)
                self.AUC(y_pred=y_pred_act, y=DX_onehot)
        # log validation metrics
        loss_results = self.loss_epoch.aggregate()
        cm_result = self.cls_metrics.aggregate()
        AUC_results = self.AUC.aggregate()
        test_res_all = [float(loss_results), float(cm_result[0]), float(cm_result[1]),
                        float(cm_result[2]), float(cm_result[3]), AUC_results]
        self.logger.print_message(f'Test - Loss:{test_res_all[0]:.4f} '
                                  f'ACC:{test_res_all[1]:.4f} '
                                  f'SEN:{test_res_all[2]:.4f} '
                                  f'SPE:{test_res_all[3]:.4f} '
                                  f'F1:{test_res_all[4]:.4f} '
                                  f'AUC:{test_res_all[5]:.4f}')
        return test_res_all

    def start_eval_vis(self, test_loader):
        self._load_checkpoint()
        self.vec_out = torch.zeros((0, self.args.dim), dtype=torch.float32)

        def hook_fn(m, i, o):
            # o -> (b, dim, h, w, d)
            o = rearrange(o, 'b c h w d -> (b h w d) c')
            self.vec_out = torch.cat((self.vec_out, o.detach().cpu()), 0)

        self.nets.CNN.conv5.register_forward_hook(hook_fn)

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # fetch images and labels
                if self.args.modality == 'MRI':
                    IMG, _, _, _, DX = data
                    IMG, DX = IMG.to(self.device), DX.to(self.device)
                elif self.args.modality == 'PET':
                    _, IMG, _, _, DX = data
                    IMG, DX = IMG.to(self.device), DX.to(self.device)
                else:
                    IMG, DX = None, None
                _, _ = self.compute_CLS_loss(IMG, DX)

        # T-sne Visualization
        tsne = TSNE(2, init='pca', learning_rate='auto')
        tsne_proj = tsne.fit_transform(self.vec_out)
        # Plot those points as a scatter plot and label them based on the pred labels
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(tsne_proj[:, 0], tsne_proj[:, 1])
        plt.savefig('a.png')


class model_CLS_CNN(nn.Module):
    def __init__(self, args, fold):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = build_models(args, 'CNN')
        # create checkpoint dir
        self.fold = fold
        self.checkpoint_dir = os.path.join(args.checkpoints_dir, args.name, str(fold))
        if not os.path.exists(self.checkpoint_dir):
            utils.mkdir(self.checkpoint_dir)
        self.logger = utils.Logger(self.checkpoint_dir)
        # print network status
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        # create optimizers and checkpoints
        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                self.optims[net] = torch.optim.AdamW(
                    params=self.nets[net].parameters(), lr=args.lr, betas=(0.9, 0.999),
                    eps=1e-08, weight_decay=args.weight_decay)
            self.ckptios = [
                utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]
        else:
            self.ckptios = [
                utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]
        # to CUDA device
        self.to(self.device)
        # Initialize network parameters
        for name, network in self.named_children():
            self.logger.print_message('Initializing %s...' % name)
            network.apply(utils.he_init)
        # transform utils
        self.post_pred = Compose([Activations(softmax=True)])
        self.post_pred_argmax = Compose([AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([AsDiscrete(to_onehot=2)])

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def _save_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.save()

    def _load_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.load()

    def compute_CLS_loss(self, MRI, PET, label):
        # forward CNN
        MRI_feats = self.nets.MRI(MRI)
        PET_feats = self.nets.PET(PET)
        # Average Pool
        MRI_feats = F.adaptive_max_pool3d(MRI_feats, 1).view(MRI.shape[0], -1)
        PET_feats = F.adaptive_max_pool3d(PET_feats, 1).view(PET.shape[0], -1)
        # forward MLP
        logits = self.nets.CLS(torch.cat([MRI_feats, PET_feats], dim=1))
        loss_CLS = F.cross_entropy(logits, label)
        return logits, loss_CLS

    def start_train(self, train_loaders, val_loader, test_loader):
        args = self.args
        # use acc & f1 score to select model
        best_acc = 0.0
        best_f1 = 0.0
        best_epoch = 1
        self.logger.print_message('Start training...')
        start_time = time.time()
        # define evaluation metrics
        loss_epoch = CumulativeAverage()
        cls_metrics = ConfusionMatrixMetric(metric_name=["accuracy", 'sensitivity', 'specificity', 'f1 score'],
                                            include_background=False, reduction='mean')
        AUC = ROCAUCMetric(average='micro')

        for epoch in range(0, args.epochs):
            # training iterations
            for name, model in self.nets.items():
                model.train()
            for i, data in enumerate(train_loaders):
                # fetch images and labels
                MRI, FDG, tabular, sub_missing, DX = data
                MRI, FDG, tabular, DX = MRI.to(self.device), FDG.to(self.device), \
                                        tabular.to(self.device), DX.to(self.device)
                logits, loss_CLS = self.compute_CLS_loss(MRI, FDG, DX)
                # backward
                self._reset_grad()
                loss_CLS.backward()
                self.optims.MRI.step()
                self.optims.PET.step()
                self.optims.CLS.step()
                # compute training metrics
                #    epoch average loss
                loss_epoch.append(loss_CLS)
                #    cls metrics
                DX_onehot = [self.post_label(i) for i in decollate_batch(DX, detach=False)]
                y_pred_act = [self.post_pred(i) for i in decollate_batch(logits)]
                y_pred_act_onehot = [self.post_pred_argmax(i) for i in decollate_batch(logits)]
                cls_metrics(y_pred=y_pred_act_onehot, y=DX_onehot)
                AUC(y_pred=y_pred_act, y=DX_onehot)
            # log training metrics
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
            self.logger.print_message(f'Epoch {epoch + 1}/{args.epochs} - Elapsed time {elapsed}')
            loss_results = loss_epoch.aggregate()
            cm_result = cls_metrics.aggregate()
            AUC_results = AUC.aggregate()
            self.logger.print_message(f'Trainng    - Loss:{float(loss_results):.4f} '
                                      f'ACC:{float(cm_result[0]):.4f} '
                                      f'SEN:{float(cm_result[1]):.4f} '
                                      f'SPE:{float(cm_result[2]):.4f} '
                                      f'F1:{float(cm_result[3]):.4f} '
                                      f'AUC:{AUC_results:.4f}')
            # reset metrics
            loss_epoch.reset()
            cls_metrics.reset()
            AUC.reset()

            # validation iterations
            for name, model in self.nets.items():
                model.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    # fetch images and labels
                    MRI, FDG, tabular, sub_missing, DX = data
                    MRI, FDG, tabular, DX = MRI.to(self.device), FDG.to(self.device), \
                                            tabular.to(self.device), DX.to(self.device)
                    logits, loss_CLS = self.compute_CLS_loss(MRI, FDG, DX)
                    # compute training metrics
                    #    epoch average loss
                    loss_epoch.append(loss_CLS)
                    #    cls metrics
                    DX_onehot = [self.post_label(i) for i in decollate_batch(DX, detach=False)]
                    y_pred_act = [self.post_pred(i) for i in decollate_batch(logits)]
                    y_pred_act_onehot = [self.post_pred_argmax(i) for i in decollate_batch(logits)]
                    cls_metrics(y_pred=y_pred_act_onehot, y=DX_onehot)
                    AUC(y_pred=y_pred_act, y=DX_onehot)
            # log validation metrics
            loss_results = loss_epoch.aggregate()
            cm_result = cls_metrics.aggregate()
            AUC_results = AUC.aggregate()
            self.logger.print_message(f'Validation - Loss:{float(loss_results):.4f} '
                                      f'ACC:{float(cm_result[0]):.4f} '
                                      f'SEN:{float(cm_result[1]):.4f} '
                                      f'SPE:{float(cm_result[2]):.4f} '
                                      f'F1:{float(cm_result[3]):.4f} '
                                      f'AUC:{AUC_results:.4f}')
            # save best model according to the validation results
            acc = float(cm_result[0].cpu())
            f1 = float(cm_result[3].cpu())
            if acc >= best_acc and f1 >= best_f1:
                self._save_checkpoint()
                best_acc = acc
                best_f1 = f1
                best_epoch = epoch + 1
            # reset metrics
            loss_epoch.reset()
            cls_metrics.reset()
            AUC.reset()

        # start test iterations
        #   load best model
        self.logger.print_message(f'\nLoad best model at epoch {best_epoch:03d}')
        self._load_checkpoint()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # fetch images and labels
                MRI, FDG, tabular, sub_missing, DX = data
                MRI, FDG, tabular, DX = MRI.to(self.device), FDG.to(self.device), \
                                        tabular.to(self.device), DX.to(self.device)
                logits, loss_CLS = self.compute_CLS_loss(MRI, FDG, DX)
                # compute training metrics
                #    epoch average loss
                loss_epoch.append(loss_CLS)
                #    cls metrics
                DX_onehot = [self.post_label(i) for i in decollate_batch(DX, detach=False)]
                y_pred_act = [self.post_pred(i) for i in decollate_batch(logits)]
                y_pred_act_onehot = [self.post_pred_argmax(i) for i in decollate_batch(logits)]
                cls_metrics(y_pred=y_pred_act_onehot, y=DX_onehot)
                AUC(y_pred=y_pred_act, y=DX_onehot)
        # log validation metrics
        loss_results = loss_epoch.aggregate()
        cm_result = cls_metrics.aggregate()
        AUC_results = AUC.aggregate()
        test_res_all = [float(loss_results), float(cm_result[0]), float(cm_result[1]),
                        float(cm_result[2]), float(cm_result[3]), AUC_results]
        self.logger.print_message(f'Test - Loss:{test_res_all[0]:.4f} '
                                  f'ACC:{test_res_all[1]:.4f} '
                                  f'SEN:{test_res_all[2]:.4f} '
                                  f'SPE:{test_res_all[3]:.4f} '
                                  f'F1:{test_res_all[4]:.4f} '
                                  f'AUC:{test_res_all[5]:.4f}')

        return test_res_all


class model_CLS_Transformer(nn.Module):
    def __init__(self, args, fold):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = build_models(args, 'Transformer')
        # create checkpoint dir
        self.fold = fold
        self.checkpoint_dir = os.path.join(args.checkpoints_dir, args.name, str(fold))
        if not os.path.exists(self.checkpoint_dir):
            utils.mkdir(self.checkpoint_dir)
        self.logger = utils.Logger(self.checkpoint_dir)
        # print network status
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        # create optimizers and checkpoints
        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                self.optims[net] = torch.optim.AdamW(
                    params=self.nets[net].parameters(), lr=args.lr, betas=(0.9, 0.999),
                    eps=1e-08, weight_decay=args.weight_decay)
            self.ckptios = [
                utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]
        else:
            self.ckptios = [
                utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]
        # to CUDA device
        self.to(self.device)
        # Initialize network parameters
        for name, network in self.named_children():
            self.logger.print_message('Initializing %s...' % name)
            network.apply(utils.he_init)
        # transform utils
        self.post_pred = Compose([Activations(softmax=True)])
        self.post_pred_argmax = Compose([AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([AsDiscrete(to_onehot=2)])

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def _save_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.save()

    def _load_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.load()

    def compute_CLS_loss(self, MRI, PET, label):
        # forward CNN
        MRI_feats = self.nets.MRI(MRI)
        PET_feats = self.nets.PET(PET)
        # Reshape to vectors
        MRI_feats = rearrange(MRI_feats, 'b c h w d -> b (h w d) c')
        PET_feats = rearrange(PET_feats, 'b c h w d -> b (h w d) c')
        # forward cross transformer
        gap = nn.Sequential(Rearrange('b n d -> b d n'), AdaptiveAvgPool1d(1), Rearrange('b d n -> b (d n)'))
        gmp = nn.Sequential(Rearrange('b n d -> b d n'), AdaptiveMaxPool1d(1), Rearrange('b d n -> b (d n)'))
        MRI_feats, PET_feats = self.nets.Trans(MRI_feats, PET_feats)
        mri_cls_avg = gap(MRI_feats)
        mri_cls_max = gmp(MRI_feats)
        pet_cls_avg = gap(PET_feats)
        pet_cls_max = gmp(PET_feats)
        cls_token = torch.cat([mri_cls_avg, pet_cls_avg, mri_cls_max, pet_cls_max], dim=1)
        # forward MLP
        logits = self.nets.CLS(cls_token)
        loss_CLS = F.cross_entropy(logits, label)
        return logits, loss_CLS

    def start_train(self, train_loaders, val_loader, test_loader):
        args = self.args
        # use acc & f1 score to select model
        best_acc = 0.0
        best_f1 = 0.0
        best_epoch = 1
        self.logger.print_message('Start training...')
        start_time = time.time()
        # define evaluation metrics
        loss_epoch = CumulativeAverage()
        cls_metrics = ConfusionMatrixMetric(metric_name=["accuracy", 'sensitivity', 'specificity', 'f1 score'],
                                            include_background=False, reduction='mean')
        AUC = ROCAUCMetric(average='micro')

        for epoch in range(0, args.epochs):
            # training iterations
            for name, model in self.nets.items():
                model.train()
            for i, data in enumerate(train_loaders):
                # fetch images and labels
                MRI, FDG, tabular, sub_missing, DX = data
                MRI, FDG, tabular, DX = MRI.to(self.device), FDG.to(self.device), \
                                        tabular.to(self.device), DX.to(self.device)
                logits, loss_CLS = self.compute_CLS_loss(MRI, FDG, DX)
                # backward
                self._reset_grad()
                loss_CLS.backward()
                self.optims.MRI.step()
                self.optims.PET.step()
                self.optims.Trans.step()
                self.optims.CLS.step()
                # compute training metrics
                #    epoch average loss
                loss_epoch.append(loss_CLS)
                #    cls metrics
                DX_onehot = [self.post_label(i) for i in decollate_batch(DX, detach=False)]
                y_pred_act = [self.post_pred(i) for i in decollate_batch(logits)]
                y_pred_act_onehot = [self.post_pred_argmax(i) for i in decollate_batch(logits)]
                cls_metrics(y_pred=y_pred_act_onehot, y=DX_onehot)
                AUC(y_pred=y_pred_act, y=DX_onehot)
            # log training metrics
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
            self.logger.print_message(f'Epoch {epoch + 1}/{args.epochs} - Elapsed time {elapsed}')
            loss_results = loss_epoch.aggregate()
            cm_result = cls_metrics.aggregate()
            AUC_results = AUC.aggregate()
            self.logger.print_message(f'Trainng    - Loss:{float(loss_results):.4f} '
                                      f'ACC:{float(cm_result[0]):.4f} '
                                      f'SEN:{float(cm_result[1]):.4f} '
                                      f'SPE:{float(cm_result[2]):.4f} '
                                      f'F1:{float(cm_result[3]):.4f} '
                                      f'AUC:{AUC_results:.4f}')
            # reset metrics
            loss_epoch.reset()
            cls_metrics.reset()
            AUC.reset()

            # validation iterations
            for name, model in self.nets.items():
                model.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    # fetch images and labels
                    MRI, FDG, tabular, sub_missing, DX = data
                    MRI, FDG, tabular, DX = MRI.to(self.device), FDG.to(self.device), \
                                            tabular.to(self.device), DX.to(self.device)
                    logits, loss_CLS = self.compute_CLS_loss(MRI, FDG, DX)
                    # compute training metrics
                    #    epoch average loss
                    loss_epoch.append(loss_CLS)
                    #    cls metrics
                    DX_onehot = [self.post_label(i) for i in decollate_batch(DX, detach=False)]
                    y_pred_act = [self.post_pred(i) for i in decollate_batch(logits)]
                    y_pred_act_onehot = [self.post_pred_argmax(i) for i in decollate_batch(logits)]
                    cls_metrics(y_pred=y_pred_act_onehot, y=DX_onehot)
                    AUC(y_pred=y_pred_act, y=DX_onehot)
            # log validation metrics
            loss_results = loss_epoch.aggregate()
            cm_result = cls_metrics.aggregate()
            AUC_results = AUC.aggregate()
            self.logger.print_message(f'Validation - Loss:{float(loss_results):.4f} '
                                      f'ACC:{float(cm_result[0]):.4f} '
                                      f'SEN:{float(cm_result[1]):.4f} '
                                      f'SPE:{float(cm_result[2]):.4f} '
                                      f'F1:{float(cm_result[3]):.4f} '
                                      f'AUC:{AUC_results:.4f}')
            # save best model according to the validation results
            acc = float(cm_result[0].cpu())
            f1 = float(cm_result[3].cpu())
            if acc >= best_acc and f1 >= best_f1:
                self._save_checkpoint()
                best_acc = acc
                best_f1 = f1
                best_epoch = epoch + 1
            # reset metrics
            loss_epoch.reset()
            cls_metrics.reset()
            AUC.reset()

        # start test iterations
        #   load best model
        self.logger.print_message(f'\nLoad best model at epoch {best_epoch:03d}')
        self._load_checkpoint()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # fetch images and labels
                MRI, FDG, tabular, sub_missing, DX = data
                MRI, FDG, tabular, DX = MRI.to(self.device), FDG.to(self.device), \
                                        tabular.to(self.device), DX.to(self.device)
                logits, loss_CLS = self.compute_CLS_loss(MRI, FDG, DX)
                # compute training metrics
                #    epoch average loss
                loss_epoch.append(loss_CLS)
                #    cls metrics
                DX_onehot = [self.post_label(i) for i in decollate_batch(DX, detach=False)]
                y_pred_act = [self.post_pred(i) for i in decollate_batch(logits)]
                y_pred_act_onehot = [self.post_pred_argmax(i) for i in decollate_batch(logits)]
                cls_metrics(y_pred=y_pred_act_onehot, y=DX_onehot)
                AUC(y_pred=y_pred_act, y=DX_onehot)
        # log validation metrics
        loss_results = loss_epoch.aggregate()
        cm_result = cls_metrics.aggregate()
        AUC_results = AUC.aggregate()
        test_res_all = [float(loss_results.cpu()), float(cm_result[0].cpu()), float(cm_result[1].cpu()),
                        float(cm_result[2].cpu()), float(cm_result[3].cpu()), AUC_results]
        self.logger.print_message(f'Test - Loss:{test_res_all[0]:.4f} '
                                  f'ACC:{test_res_all[1]:.4f} '
                                  f'SEN:{test_res_all[2]:.4f} '
                                  f'SPE:{test_res_all[3]:.4f} '
                                  f'F1:{test_res_all[4]:.4f} '
                                  f'AUC:{test_res_all[5]:.4f}')

        return test_res_all


class model_CLS_VQCNN_Single(nn.Module):
    def __init__(self, args, fold):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = build_models(args, 'VQCNN_Single')
        # create checkpoint dir
        self.fold = fold
        self.checkpoint_dir = os.path.join(args.checkpoints_dir, args.name, str(fold))
        if not os.path.exists(self.checkpoint_dir):
            utils.mkdir(self.checkpoint_dir)
        if args.modality == 'MRI':
            self.logger = utils.Logger(self.checkpoint_dir, 'mri_log.txt')
        elif args.modality == 'PET':
            self.logger = utils.Logger(self.checkpoint_dir, 'pet_log.txt')
        # print network status
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        # create optimizers and checkpoints
        self.optims = Munch()
        for net in self.nets.keys():
            if net == 'CODEBOOK':
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(), lr=args.lr * 100, betas=(0.9, 0.999), eps=1e-08)
            else:
                self.optims[net] = torch.optim.AdamW(
                    params=self.nets[net].parameters(), lr=args.lr, betas=(0.9, 0.999),
                    eps=1e-08, weight_decay=args.weight_decay)
        if args.modality == 'MRI':
            self.ckptios = [
                utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_MRI_nets.ckpt'), **self.nets)]
        elif args.modality == 'PET':
            self.ckptios = [
                utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_PET_nets.ckpt'), **self.nets)]
        else:
            self.ckptios = None
        # to CUDA device
        self.to(self.device)
        # Initialize network parameters
        for name, network in self.named_children():
            if name == 'CODEBOOK':
                continue
            self.logger.print_message('Initializing %s...' % name)
            network.apply(utils.he_init)
        # transform utils
        self.post_pred = Compose([Activations(softmax=True)])
        self.post_pred_argmax = Compose([AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([AsDiscrete(to_onehot=2)])
        # define evaluation metrics
        self.loss_CLS_epoch = CumulativeAverage()
        self.loss_Q_epoch = CumulativeAverage()
        self.cls_metrics = ConfusionMatrixMetric(metric_name=["accuracy", 'sensitivity', 'specificity', 'f1 score'],
                                                 include_background=False, reduction='mean')
        self.AUC = ROCAUCMetric(average='micro')

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def _save_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.save()

    def _load_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.load()

    def compute_CLS_loss(self, IMG, label):
        # forward CNN
        feats = self.nets.CNN(IMG)
        feats = rearrange(feats, 'b c h w d -> b (h w d) c')
        # quant feats
        codebook_mapping, codebook_indices, loss_Q = self.nets.CODEBOOK(feats)  # indices -> [bz, 3, 4, 3]
        # Average Pool
        codebook_mapping = rearrange(codebook_mapping, 'b n c -> b c n')
        feats = F.adaptive_max_pool1d(codebook_mapping, 1).view(IMG.shape[0], -1)
        # forward MLP
        logits = self.nets.CLS(feats)
        loss_CLS = F.cross_entropy(logits, label)
        return logits, codebook_indices, loss_CLS, loss_Q

    def compute_CLS_loss_withoutVQ(self, IMG, label):
        # forward CNN
        feats = self.nets.CNN(IMG)
        # Average Pool
        feats = F.adaptive_max_pool3d(feats, 1).view(IMG.shape[0], -1)
        # forward MLP
        logits = self.nets.CLS(feats)
        loss_CLS = F.cross_entropy(logits, label)
        return logits, loss_CLS, loss_CLS

    def one_epoch(self, loader, status, epoch, args):
        if status == 'train':
            for name, model in self.nets.items():
                model.train()
        if status == 'eval' or status == 'test':
            for name, model in self.nets.items():
                model.eval()
        if status == 'test':
            all_indices = []
        # start epoch
        for i, data in enumerate(loader):
            # fetch images and labels
            if args.modality == 'MRI':
                IMG, _, _, _, DX = data
                IMG, DX = IMG.to(self.device), DX.to(self.device)
            elif args.modality == 'PET':
                _, IMG, _, _, DX = data
                IMG, DX = IMG.to(self.device), DX.to(self.device)
            else:
                IMG, DX = None, None
            # forward
            if epoch >= args.warmup_epochs:
                logits, indices, loss_CLS, loss_Q = self.compute_CLS_loss(IMG, DX)
                loss = loss_CLS + loss_Q
                if status == 'test':
                    all_indices.append(indices)
            else:
                logits, loss_CLS, loss_Q = self.compute_CLS_loss_withoutVQ(IMG, DX)
                loss = loss_CLS
                indices = None
            # backward if training
            if status == 'train':
                self._reset_grad()
                loss.backward()
                self.optims.CNN.step()
                self.optims.CLS.step()
                self.optims.CODEBOOK.step()
            #    epoch average loss
            self.loss_CLS_epoch.append(loss_CLS)
            self.loss_Q_epoch.append(loss_Q)
            #    cls metrics
            DX_onehot = [self.post_label(i) for i in decollate_batch(DX, detach=False)]
            y_pred_act = [self.post_pred(i) for i in decollate_batch(logits)]
            y_pred_act_onehot = [self.post_pred_argmax(i) for i in decollate_batch(logits)]
            self.cls_metrics(y_pred=y_pred_act_onehot, y=DX_onehot)
            self.AUC(y_pred=y_pred_act, y=DX_onehot)
        # logging metrics
        loss_CLS_results = self.loss_CLS_epoch.aggregate()
        loss_Q_results = self.loss_Q_epoch.aggregate()
        cm_result = self.cls_metrics.aggregate()
        AUC_results = self.AUC.aggregate()
        if status == 'train':
            elapsed = time.time() - self.start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
            self.logger.print_message(f'Epoch {epoch + 1}/{args.epochs} - Elapsed time {elapsed}')
        self.logger.print_message(f'{status}    - Loss_CLS:{float(loss_CLS_results):.4f} '
                                  f'Loss_Q:{float(loss_Q_results):.4f} '
                                  f'ACC:{float(cm_result[0]):.4f} '
                                  f'SEN:{float(cm_result[1]):.4f} '
                                  f'SPE:{float(cm_result[2]):.4f} '
                                  f'F1:{float(cm_result[3]):.4f} '
                                  f'AUC:{AUC_results:.4f}')
        # save best model according to the validation results
        if status == 'eval':
            if epoch >= args.warmup_epochs:
                acc = float(cm_result[0].cpu())
                f1 = float(cm_result[3].cpu())
                if acc >= self.best_acc and f1 >= self.best_f1:
                    self._save_checkpoint()
                    self.best_acc = acc
                    self.best_f1 = f1
                    self.best_epoch = epoch + 1
        # reset metrics
        self.loss_CLS_epoch.reset()
        self.loss_Q_epoch.reset()
        self.cls_metrics.reset()
        self.AUC.reset()
        metric_res = [float(loss_CLS_results), float(loss_Q_results), float(cm_result[0]),
                      float(cm_result[1]), float(cm_result[2]), float(cm_result[3]), AUC_results]
        if status == 'test':
            metric_res = (metric_res, all_indices)
        return metric_res

    def start_train(self, train_loader, val_loader):
        args = self.args
        # use acc & f1 score to select model
        self.best_acc = 0.0
        self.best_f1 = 0.0
        self.best_epoch = 1
        self.logger.print_message('Start training...')
        self.start_time = time.time()

        for epoch in range(0, args.epochs):
            # training iterations
            _ = self.one_epoch(train_loader, 'train', epoch, args)
            # validation iterations
            with torch.no_grad():
                _ = self.one_epoch(val_loader, 'eval', epoch, args)

    def start_test(self, test_loader):
        args = self.args
        # start test iterations
        # #   load best model
        # self.logger.print_message(f'\nLoad best model at epoch {self.best_epoch:03d}')
        # start test iterations
        #   load best model
        self._load_checkpoint()
        #   test epoch
        with torch.no_grad():
            test_res, all_indices = self.one_epoch(test_loader, 'test', args.epochs, args)
        # analyze the indices
        # all_indices = torch.cat(all_indices, dim=0).cpu().numpy()  # (num, 3, 4, 3)
        # indices, indices_counts = np.unique(all_indices, return_counts=True)
        # print(indices)
        # print(indices_counts)
        return test_res


class model_CLS_VQCNN(nn.Module):
    def __init__(self, args, fold):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = build_models(args, 'VQCNN')
        # create checkpoint dir
        self.fold = fold
        self.checkpoint_dir = os.path.join(args.checkpoints_dir, args.name, str(fold))
        if not os.path.exists(self.checkpoint_dir):
            utils.mkdir(self.checkpoint_dir)
        self.logger = utils.Logger(self.checkpoint_dir)
        # print network status
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        # create optimizers and checkpoints
        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                if net == 'MRICODEBOOK' or net == 'PETCODEBOOK':
                    self.optims[net] = torch.optim.AdamW(
                        params=self.nets[net].parameters(), lr=args.lr / 10, betas=(0.9, 0.999),
                        eps=1e-08, weight_decay=args.weight_decay)
                else:
                    self.optims[net] = torch.optim.AdamW(
                        params=self.nets[net].parameters(), lr=args.lr, betas=(0.9, 0.999),
                        eps=1e-08, weight_decay=args.weight_decay)
            self.ckptios = [
                utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]
        else:
            self.ckptios = [
                utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]
        # to CUDA device
        self.to(self.device)
        # Initialize network parameters
        for name, network in self.named_children():
            self.logger.print_message('Initializing %s...' % name)
            network.apply(utils.he_init)
        # transform utils
        self.post_pred = Compose([Activations(softmax=True)])
        self.post_pred_argmax = Compose([AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([AsDiscrete(to_onehot=2)])

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def _save_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.save()

    def _load_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.load()

    def compute_CLS_loss(self, MRI, PET, label):
        # forward CNN
        MRI_feats = self.nets.MRI(MRI)
        PET_feats = self.nets.PET(PET)
        # quant feats
        MRI_quant_feats, _, loss_Q_MRI = self.nets.MRICODEBOOK(MRI_feats)
        PET_quant_feats, _, loss_Q_PET = self.nets.PETCODEBOOK(PET_feats)
        # Average Pool
        MRI_feats = F.adaptive_max_pool3d(MRI_quant_feats, 1).view(MRI.shape[0], -1)
        PET_feats = F.adaptive_max_pool3d(PET_quant_feats, 1).view(PET.shape[0], -1)
        # forward MLP
        logits = self.nets.CLS(torch.cat([MRI_feats, PET_feats], dim=1))
        loss_CLS = F.cross_entropy(logits, label)
        return logits, loss_CLS, loss_Q_MRI, loss_Q_PET

    def start_train(self, train_loaders, val_loader, test_loader):
        args = self.args
        # use acc & f1 score to select model
        best_acc = 0.0
        best_f1 = 0.0
        best_epoch = 1
        self.logger.print_message('Start training...')
        start_time = time.time()
        # define evaluation metrics
        loss_CLS_epoch = CumulativeAverage()
        loss_MRI_Q_epoch = CumulativeAverage()
        loss_PET_Q_epoch = CumulativeAverage()
        cls_metrics = ConfusionMatrixMetric(metric_name=["accuracy", 'sensitivity', 'specificity', 'f1 score'],
                                            include_background=False, reduction='mean')
        AUC = ROCAUCMetric(average='micro')

        for epoch in range(0, args.epochs):
            # training iterations
            for name, model in self.nets.items():
                model.train()
            for i, data in enumerate(train_loaders):
                # fetch images and labels
                MRI, FDG, _, _, DX = data
                MRI, FDG, DX = MRI.to(self.device), FDG.to(self.device), DX.to(self.device)
                logits, loss_CLS, loss_MRI_Q, loss_PET_Q = self.compute_CLS_loss(MRI, FDG, DX)
                loss = loss_CLS + loss_MRI_Q + loss_PET_Q
                # backward
                self._reset_grad()
                loss.backward()
                self.optims.MRI.step()
                self.optims.PET.step()
                self.optims.MRICODEBOOK.step()
                self.optims.PETCODEBOOK.step()
                self.optims.CLS.step()
                # compute training metrics
                #    epoch average loss
                loss_CLS_epoch.append(loss_CLS)
                loss_MRI_Q_epoch.append(loss_MRI_Q)
                loss_PET_Q_epoch.append(loss_PET_Q)
                #    cls metrics
                DX_onehot = [self.post_label(i) for i in decollate_batch(DX, detach=False)]
                y_pred_act = [self.post_pred(i) for i in decollate_batch(logits)]
                y_pred_act_onehot = [self.post_pred_argmax(i) for i in decollate_batch(logits)]
                cls_metrics(y_pred=y_pred_act_onehot, y=DX_onehot)
                AUC(y_pred=y_pred_act, y=DX_onehot)
            # log training metrics
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
            self.logger.print_message(f'Epoch {epoch + 1}/{args.epochs} - Elapsed time {elapsed}')
            loss_CLS_results = loss_CLS_epoch.aggregate()
            loss_MRI_Q_results = loss_MRI_Q_epoch.aggregate()
            loss_PET_Q_results = loss_PET_Q_epoch.aggregate()
            cm_result = cls_metrics.aggregate()
            AUC_results = AUC.aggregate()
            self.logger.print_message(f'Trainng    - Loss_CLS:{float(loss_CLS_results):.4f} '
                                      f'Loss_MRI_Q:{float(loss_MRI_Q_results):.4f} '
                                      f'Loss_PET_Q:{float(loss_PET_Q_results):.4f} '
                                      f'ACC:{float(cm_result[0].cpu()):.4f} '
                                      f'SEN:{float(cm_result[1].cpu()):.4f} '
                                      f'SPE:{float(cm_result[2].cpu()):.4f} '
                                      f'F1:{float(cm_result[3].cpu()):.4f} '
                                      f'AUC:{AUC_results:.4f}')
            # reset metrics
            loss_CLS_epoch.reset()
            loss_MRI_Q_epoch.reset()
            loss_PET_Q_epoch.reset()
            cls_metrics.reset()
            AUC.reset()

            # validation iterations
            for name, model in self.nets.items():
                model.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    # fetch images and labels
                    MRI, FDG, _, _, DX = data
                    MRI, FDG, DX = MRI.to(self.device), FDG.to(self.device), DX.to(self.device)
                    logits, loss_CLS, loss_MRI_Q, loss_PET_Q = self.compute_CLS_loss(MRI, FDG, DX)
                    # compute training metrics
                    #    epoch average loss
                    loss_CLS_epoch.append(loss_CLS)
                    loss_MRI_Q_epoch.append(loss_MRI_Q)
                    loss_PET_Q_epoch.append(loss_PET_Q)
                    #    cls metrics
                    DX_onehot = [self.post_label(i) for i in decollate_batch(DX, detach=False)]
                    y_pred_act = [self.post_pred(i) for i in decollate_batch(logits)]
                    y_pred_act_onehot = [self.post_pred_argmax(i) for i in decollate_batch(logits)]
                    cls_metrics(y_pred=y_pred_act_onehot, y=DX_onehot)
                    AUC(y_pred=y_pred_act, y=DX_onehot)
            # log validation metrics
            loss_CLS_results = loss_CLS_epoch.aggregate()
            loss_MRI_Q_results = loss_MRI_Q_epoch.aggregate()
            loss_PET_Q_results = loss_PET_Q_epoch.aggregate()
            cm_result = cls_metrics.aggregate()
            AUC_results = AUC.aggregate()
            self.logger.print_message(f'Evaluation    - Loss_CLS:{float(loss_CLS_results):.4f} '
                                      f'Loss_MRI_Q:{float(loss_MRI_Q_results):.4f} '
                                      f'Loss_PET_Q:{float(loss_PET_Q_results):.4f} '
                                      f'ACC:{float(cm_result[0].cpu()):.4f} '
                                      f'SEN:{float(cm_result[1].cpu()):.4f} '
                                      f'SPE:{float(cm_result[2].cpu()):.4f} '
                                      f'F1:{float(cm_result[3].cpu()):.4f} '
                                      f'AUC:{AUC_results:.4f}')
            # save best model according to the validation results
            acc = float(cm_result[0].cpu())
            f1 = float(cm_result[3].cpu())
            if acc >= best_acc and f1 >= best_f1:
                self._save_checkpoint()
                best_acc = acc
                best_f1 = f1
                best_epoch = epoch + 1
            # reset metrics
            loss_CLS_epoch.reset()
            loss_MRI_Q_epoch.reset()
            loss_PET_Q_epoch.reset()
            cls_metrics.reset()
            AUC.reset()

        # start test iterations
        #   load best model
        self.logger.print_message(f'\nLoad best model at epoch {best_epoch:03d}')
        self._load_checkpoint()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # fetch images and labels
                MRI, FDG, _, _, DX = data
                MRI, FDG, DX = MRI.to(self.device), FDG.to(self.device), DX.to(self.device)
                logits, loss_CLS, loss_MRI_Q, loss_PET_Q = self.compute_CLS_loss(MRI, FDG, DX)
                # compute training metrics
                #    epoch average loss
                loss_CLS_epoch.append(loss_CLS)
                loss_MRI_Q_epoch.append(loss_MRI_Q)
                loss_PET_Q_epoch.append(loss_PET_Q)
                #    cls metrics
                DX_onehot = [self.post_label(i) for i in decollate_batch(DX, detach=False)]
                y_pred_act = [self.post_pred(i) for i in decollate_batch(logits)]
                y_pred_act_onehot = [self.post_pred_argmax(i) for i in decollate_batch(logits)]
                cls_metrics(y_pred=y_pred_act_onehot, y=DX_onehot)
                AUC(y_pred=y_pred_act, y=DX_onehot)
        # log validation metrics   ``
        loss_CLS_results = loss_CLS_epoch.aggregate()
        loss_MRI_Q_results = loss_MRI_Q_epoch.aggregate()
        loss_PET_Q_results = loss_PET_Q_epoch.aggregate()
        cm_result = cls_metrics.aggregate()
        AUC_results = AUC.aggregate()
        test_res_all = [float(loss_CLS_results), float(loss_MRI_Q_results), float(loss_PET_Q_results),
                        float(cm_result[0].cpu()), float(cm_result[1].cpu()),
                        float(cm_result[2].cpu()), float(cm_result[3].cpu()), AUC_results]
        self.logger.print_message(f'Test - Loss_CLS:{test_res_all[0]:.4f} '
                                  f'Loss_MRI_Q:{test_res_all[1]:.4f} '
                                  f'Loss_PET_Q:{test_res_all[2]:.4f} '
                                  f'ACC:{test_res_all[3]:.4f} '
                                  f'SEN:{test_res_all[4]:.4f} '
                                  f'SPE:{test_res_all[5]:.4f} '
                                  f'F1:{test_res_all[6]:.4f} '
                                  f'AUC:{test_res_all[7]:.4f}')

        return test_res_all


class model_CLS_VQCNN_Transformer(nn.Module):
    def __init__(self, args, fold):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = build_models(args, 'VQTransformer')
        # create checkpoint dir
        self.fold = fold
        self.checkpoint_dir = os.path.join(args.checkpoints_dir, args.name, str(fold))
        if not os.path.exists(self.checkpoint_dir):
            utils.mkdir(self.checkpoint_dir)
        self.logger = utils.Logger(self.checkpoint_dir)
        # print network status
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        # create optimizers and checkpoints
        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                if net == 'MRICODEBOOK' or net == 'PETCODEBOOK':
                    self.optims[net] = torch.optim.AdamW(
                        params=self.nets[net].parameters(), lr=args.lr / 10, betas=(0.9, 0.999),
                        eps=1e-08, weight_decay=args.weight_decay)
                else:
                    self.optims[net] = torch.optim.AdamW(
                        params=self.nets[net].parameters(), lr=args.lr, betas=(0.9, 0.999),
                        eps=1e-08, weight_decay=args.weight_decay)
            self.ckptios = [
                utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]
        else:
            self.ckptios = [
                utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]
        # to CUDA device
        self.to(self.device)
        # Initialize network parameters
        for name, network in self.named_children():
            self.logger.print_message('Initializing %s...' % name)
            network.apply(utils.he_init)
        # transform utils
        self.post_pred = Compose([Activations(softmax=True)])
        self.post_pred_argmax = Compose([AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([AsDiscrete(to_onehot=2)])

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def _save_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.save()

    def _load_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.load()

    def compute_CLS_loss(self, MRI, PET, label):
        # forward CNN
        MRI_feats = self.nets.MRI(MRI)
        PET_feats = self.nets.PET(PET)
        # quant feats
        MRI_quant_feats, _, loss_Q_MRI = self.nets.MRICODEBOOK(MRI_feats)
        PET_quant_feats, _, loss_Q_PET = self.nets.PETCODEBOOK(PET_feats)
        # Reshape to vectors
        MRI_feats = rearrange(MRI_quant_feats, 'b c h w d -> b (h w d) c')
        PET_feats = rearrange(PET_quant_feats, 'b c h w d -> b (h w d) c')
        # forward cross transformer
        gap = nn.Sequential(Rearrange('b n d -> b d n'), AdaptiveAvgPool1d(1), Rearrange('b d n -> b (d n)'))
        gmp = nn.Sequential(Rearrange('b n d -> b d n'), AdaptiveMaxPool1d(1), Rearrange('b d n -> b (d n)'))
        MRI_feats, PET_feats = self.nets.Trans(MRI_feats, PET_feats)
        mri_cls_avg = gap(MRI_feats)
        mri_cls_max = gmp(MRI_feats)
        pet_cls_avg = gap(PET_feats)
        pet_cls_max = gmp(PET_feats)
        cls_token = torch.cat([mri_cls_avg, pet_cls_avg, mri_cls_max, pet_cls_max], dim=1)
        # forward MLP
        logits = self.nets.CLS(cls_token)
        loss_CLS = F.cross_entropy(logits, label)
        return logits, loss_CLS, loss_Q_MRI, loss_Q_PET

    def start_train(self, train_loaders, val_loader, test_loader):
        args = self.args
        # use acc & f1 score to select model
        best_acc = 0.0
        best_f1 = 0.0
        best_epoch = 1
        self.logger.print_message('Start training...')
        start_time = time.time()
        # define evaluation metrics
        loss_CLS_epoch = CumulativeAverage()
        loss_MRI_Q_epoch = CumulativeAverage()
        loss_PET_Q_epoch = CumulativeAverage()
        cls_metrics = ConfusionMatrixMetric(metric_name=["accuracy", 'sensitivity', 'specificity', 'f1 score'],
                                            include_background=False, reduction='mean')
        AUC = ROCAUCMetric(average='micro')

        for epoch in range(0, args.epochs):
            # training iterations
            for name, model in self.nets.items():
                model.train()
            for i, data in enumerate(train_loaders):
                # fetch images and labels
                MRI, FDG, _, _, DX = data
                MRI, FDG, DX = MRI.to(self.device), FDG.to(self.device), DX.to(self.device)
                logits, loss_CLS, loss_MRI_Q, loss_PET_Q = self.compute_CLS_loss(MRI, FDG, DX)
                loss = loss_CLS + loss_MRI_Q + loss_PET_Q
                # backward
                self._reset_grad()
                loss.backward()
                self.optims.MRI.step()
                self.optims.PET.step()
                self.optims.MRICODEBOOK.step()
                self.optims.PETCODEBOOK.step()
                self.optims.Trans.step()
                self.optims.CLS.step()
                # compute training metrics
                #    epoch average loss
                loss_CLS_epoch.append(loss_CLS)
                loss_MRI_Q_epoch.append(loss_MRI_Q)
                loss_PET_Q_epoch.append(loss_PET_Q)
                #    cls metrics
                DX_onehot = [self.post_label(i) for i in decollate_batch(DX, detach=False)]
                y_pred_act = [self.post_pred(i) for i in decollate_batch(logits)]
                y_pred_act_onehot = [self.post_pred_argmax(i) for i in decollate_batch(logits)]
                cls_metrics(y_pred=y_pred_act_onehot, y=DX_onehot)
                AUC(y_pred=y_pred_act, y=DX_onehot)
            # log training metrics
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
            self.logger.print_message(f'Epoch {epoch + 1}/{args.epochs} - Elapsed time {elapsed}')
            loss_CLS_results = loss_CLS_epoch.aggregate()
            loss_MRI_Q_results = loss_MRI_Q_epoch.aggregate()
            loss_PET_Q_results = loss_PET_Q_epoch.aggregate()
            cm_result = cls_metrics.aggregate()
            AUC_results = AUC.aggregate()
            self.logger.print_message(f'Trainng    - Loss_CLS:{float(loss_CLS_results):.4f} '
                                      f'Loss_MRI_Q:{float(loss_MRI_Q_results):.4f} '
                                      f'Loss_PET_Q:{float(loss_PET_Q_results):.4f} '
                                      f'ACC:{float(cm_result[0].cpu()):.4f} '
                                      f'SEN:{float(cm_result[1].cpu()):.4f} '
                                      f'SPE:{float(cm_result[2].cpu()):.4f} '
                                      f'F1:{float(cm_result[3].cpu()):.4f} '
                                      f'AUC:{AUC_results:.4f}')
            # reset metrics
            loss_CLS_epoch.reset()
            loss_MRI_Q_epoch.reset()
            loss_PET_Q_epoch.reset()
            cls_metrics.reset()
            AUC.reset()

            # validation iterations
            for name, model in self.nets.items():
                model.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    # fetch images and labels
                    MRI, FDG, _, _, DX = data
                    MRI, FDG, DX = MRI.to(self.device), FDG.to(self.device), DX.to(self.device)
                    logits, loss_CLS, loss_MRI_Q, loss_PET_Q = self.compute_CLS_loss(MRI, FDG, DX)
                    # compute training metrics
                    #    epoch average loss
                    loss_CLS_epoch.append(loss_CLS)
                    loss_MRI_Q_epoch.append(loss_MRI_Q)
                    loss_PET_Q_epoch.append(loss_PET_Q)
                    #    cls metrics
                    DX_onehot = [self.post_label(i) for i in decollate_batch(DX, detach=False)]
                    y_pred_act = [self.post_pred(i) for i in decollate_batch(logits)]
                    y_pred_act_onehot = [self.post_pred_argmax(i) for i in decollate_batch(logits)]
                    cls_metrics(y_pred=y_pred_act_onehot, y=DX_onehot)
                    AUC(y_pred=y_pred_act, y=DX_onehot)
            # log validation metrics
            loss_CLS_results = loss_CLS_epoch.aggregate()
            loss_MRI_Q_results = loss_MRI_Q_epoch.aggregate()
            loss_PET_Q_results = loss_PET_Q_epoch.aggregate()
            cm_result = cls_metrics.aggregate()
            AUC_results = AUC.aggregate()
            self.logger.print_message(f'Evaluation    - Loss_CLS:{float(loss_CLS_results):.4f} '
                                      f'Loss_MRI_Q:{float(loss_MRI_Q_results):.4f} '
                                      f'Loss_PET_Q:{float(loss_PET_Q_results):.4f} '
                                      f'ACC:{float(cm_result[0].cpu()):.4f} '
                                      f'SEN:{float(cm_result[1].cpu()):.4f} '
                                      f'SPE:{float(cm_result[2].cpu()):.4f} '
                                      f'F1:{float(cm_result[3].cpu()):.4f} '
                                      f'AUC:{AUC_results:.4f}')
            # save best model according to the validation results
            acc = float(cm_result[0].cpu())
            f1 = float(cm_result[3].cpu())
            if acc >= best_acc and f1 >= best_f1:
                self._save_checkpoint()
                best_acc = acc
                best_f1 = f1
                best_epoch = epoch + 1
            # reset metrics
            loss_CLS_epoch.reset()
            loss_MRI_Q_epoch.reset()
            loss_PET_Q_epoch.reset()
            cls_metrics.reset()
            AUC.reset()

        # start test iterations
        #   load best model
        self.logger.print_message(f'\nLoad best model at epoch {best_epoch:03d}')
        self._load_checkpoint()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # fetch images and labels
                MRI, FDG, _, _, DX = data
                MRI, FDG, DX = MRI.to(self.device), FDG.to(self.device), DX.to(self.device)
                logits, loss_CLS, loss_MRI_Q, loss_PET_Q = self.compute_CLS_loss(MRI, FDG, DX)
                # compute training metrics
                #    epoch average loss
                loss_CLS_epoch.append(loss_CLS)
                loss_MRI_Q_epoch.append(loss_MRI_Q)
                loss_PET_Q_epoch.append(loss_PET_Q)
                #    cls metrics
                DX_onehot = [self.post_label(i) for i in decollate_batch(DX, detach=False)]
                y_pred_act = [self.post_pred(i) for i in decollate_batch(logits)]
                y_pred_act_onehot = [self.post_pred_argmax(i) for i in decollate_batch(logits)]
                cls_metrics(y_pred=y_pred_act_onehot, y=DX_onehot)
                AUC(y_pred=y_pred_act, y=DX_onehot)
        # log validation metrics
        loss_CLS_results = loss_CLS_epoch.aggregate()
        loss_MRI_Q_results = loss_MRI_Q_epoch.aggregate()
        loss_PET_Q_results = loss_PET_Q_epoch.aggregate()
        cm_result = cls_metrics.aggregate()
        AUC_results = AUC.aggregate()
        test_res_all = [float(loss_CLS_results), float(loss_MRI_Q_results), float(loss_PET_Q_results),
                        float(cm_result[0].cpu()), float(cm_result[1].cpu()),
                        float(cm_result[2].cpu()), float(cm_result[3].cpu()), AUC_results]
        self.logger.print_message(f'Test - Loss_CLS:{test_res_all[0]:.4f} '
                                  f'Loss_MRI_Q:{test_res_all[1]:.4f} '
                                  f'Loss_PET_Q:{test_res_all[2]:.4f} '
                                  f'ACC:{test_res_all[3]:.4f} '
                                  f'SEN:{test_res_all[4]:.4f} '
                                  f'SPE:{test_res_all[5]:.4f} '
                                  f'F1:{test_res_all[6]:.4f} '
                                  f'AUC:{test_res_all[7]:.4f}')

        return test_res_all


class model_CLS_VQCNN_MULTIMODEL(nn.Module):
    def __init__(self, args, fold):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = build_models(args, 'VQCNN')
        # create checkpoint dir
        self.fold = fold
        self.checkpoint_dir = os.path.join(args.checkpoints_dir, args.name, str(fold))
        if not os.path.exists(self.checkpoint_dir):
            utils.mkdir(self.checkpoint_dir)
        self.logger = utils.Logger(self.checkpoint_dir)
        # print network status
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        # create optimizers and checkpoints
        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                if net == 'MRICODEBOOK' or net == 'PETCODEBOOK':
                    self.optims[net] = torch.optim.Adam(
                        params=self.nets[net].parameters(), lr=args.lr * 100, betas=(0.9, 0.999), eps=1e-08)
                else:
                    self.optims[net] = torch.optim.AdamW(
                        params=self.nets[net].parameters(), lr=args.lr, betas=(0.9, 0.999),
                        eps=1e-08, weight_decay=args.weight_decay)
            self.ckptios = [
                utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]
        else:
            self.ckptios = [
                utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **self.nets)]
        # to CUDA device
        self.to(self.device)
        # Initialize network parameters
        for name, network in self.named_children():
            # skip codebook initialization
            if name == 'MRICODEBOOK' or name == 'PETCODEBOOK':
                continue
            self.logger.print_message('Initializing %s...' % name)
            network.apply(utils.he_init)
        # transform utils
        self.post_pred = Compose([Activations(softmax=True)])
        self.post_pred_argmax = Compose([AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([AsDiscrete(to_onehot=2)])
        # define evaluation metrics
        self.loss_CLS_epoch = CumulativeAverage()
        self.loss_Q_epoch = CumulativeAverage()
        self.loss_Matching_epoch = CumulativeAverage()
        self.cls_metrics = ConfusionMatrixMetric(metric_name=["accuracy", 'sensitivity', 'specificity', 'f1 score'],
                                                 include_background=False, reduction='mean')
        self.AUC = ROCAUCMetric(average='micro')

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def _save_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.save()

    def _load_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.load()

    def compute_loss(self, MRI, PET, label):
        # forward CNN
        MRI_feats = self.nets.MRI(MRI)
        PET_feats = self.nets.PET(PET)
        # quant feats
        MRI_feats = rearrange(MRI_feats, 'b c h w d -> b (h w d) c')
        PET_feats = rearrange(PET_feats, 'b c h w d -> b (h w d) c')
        MRI_quant_feats, MRI_codebook_status, loss_Q_MRI = self.nets.MRICODEBOOK(MRI_feats.detach(), return_prob=True)
        PET_quant_feats, PET_codebook_status, loss_Q_PET = self.nets.PETCODEBOOK(PET_feats, return_prob=True)
        MRI_indices, MRI_prob = MRI_codebook_status
        PET_indices, PET_prob = PET_codebook_status
        # quantization loss
        loss_Q = (loss_Q_MRI + loss_Q_PET) / 2
        # code matching loss
        #    CrossEntropy Loss
        loss_Matching = -(torch.sum(MRI_prob * torch.log(PET_prob)) + torch.sum(PET_prob * torch.log(MRI_prob)))
        #    KL divergence Loss
        # loss_Matching = F.kl_div(PET_prob.log(), MRI_prob, reduction='batchmean')
        #    JS divergence Loss
        # loss_Matching =

        # classification loss
        MRI_quant_feats = rearrange(MRI_quant_feats, 'b n c -> b c n')
        PET_quant_feats = rearrange(PET_quant_feats, 'b n c -> b c n')
        MRI_quant_feats = F.adaptive_max_pool1d(MRI_quant_feats, 1).view(MRI.shape[0], -1)
        PET_quant_feats = F.adaptive_max_pool1d(PET_quant_feats, 1).view(PET.shape[0], -1)
        logits = self.nets.CLS(torch.cat([MRI_quant_feats, PET_quant_feats], dim=1))
        loss_CLS = F.cross_entropy(logits, label)
        return logits, loss_CLS, loss_Q, loss_Matching

    def compute_loss_withoutVQ(self, MRI, PET, label):
        # forward CNN
        MRI_feats = self.nets.MRI(MRI)
        PET_feats = self.nets.PET(PET)
        # Average Pool
        MRI_feats = F.adaptive_max_pool3d(MRI_feats, 1).view(MRI.shape[0], -1)
        PET_feats = F.adaptive_max_pool3d(PET_feats, 1).view(PET.shape[0], -1)
        # forward MLP
        logits = self.nets.CLS(torch.cat([MRI_feats, PET_feats], dim=1))
        loss_CLS = F.cross_entropy(logits, label)
        return logits, loss_CLS, loss_CLS, loss_CLS

    def forward_test(self, MRI, label):
        # forward CNN
        MRI_feats = self.nets.MRI(MRI)
        # quant feats
        MRI_feats = rearrange(MRI_feats, 'b c h w d -> b (h w d) c')
        MRI_quant_feats, MRI_codebook_status, loss_Q_MRI = self.nets.MRICODEBOOK(MRI_feats)
        PET_quant_feats = self.nets.PETCODEBOOK.get_vec_from_indice(MRI_codebook_status)
        PET_quant_feats = PET_quant_feats.view(MRI_quant_feats.shape)
        # classifictaion
        MRI_quant_feats = rearrange(MRI_quant_feats, 'b n c -> b c n')
        PET_quant_feats = rearrange(PET_quant_feats, 'b n c -> b c n')
        MRI_quant_feats = F.adaptive_max_pool1d(MRI_quant_feats, 1).view(MRI.shape[0], -1)
        PET_quant_feats = F.adaptive_max_pool1d(PET_quant_feats, 1).view(MRI.shape[0], -1)
        logits = self.nets.CLS(torch.cat([MRI_quant_feats, PET_quant_feats], dim=1))
        loss_CLS = F.cross_entropy(logits, label)
        return logits, loss_CLS, loss_Q_MRI, loss_Q_MRI

    def one_epoch(self, loader, status, epoch, args):
        if status == 'train':
            for name, model in self.nets.items():
                model.train()
        if status == 'eval' or status == 'test':
            for name, model in self.nets.items():
                model.eval()
        # start epoch
        for i, data in enumerate(loader):
            # fetch images and labels
            MRI, FDG, _, _, DX = data
            MRI, FDG, DX = MRI.to(self.device), FDG.to(self.device), DX.to(self.device)
            # forward
            if status == 'test':
                logits, loss_CLS, loss_Q, loss_Matching = self.forward_test(MRI, DX)
            else:
                if epoch >= args.warmup_epochs:
                    logits, loss_CLS, loss_Q, loss_Matching = self.compute_loss(MRI, FDG, DX)
                    loss = loss_CLS + loss_Q + loss_Matching
                else:
                    logits, loss_CLS, loss_Q, loss_Matching = self.compute_loss_withoutVQ(MRI, FDG, DX)
                    loss = loss_CLS
            # backward if training
            if status == 'train':
                # cls loss
                self._reset_grad()
                loss.backward()
                self.optims.MRI.step()
                self.optims.PET.step()
                self.optims.CLS.step()
                self.optims.PETCODEBOOK.step()
                self.optims.MRICODEBOOK.step()

            #    epoch average loss
            self.loss_CLS_epoch.append(loss_CLS)
            self.loss_Q_epoch.append(loss_Q)
            self.loss_Matching_epoch.append(loss_Matching)
            #    cls metrics
            DX_onehot = [self.post_label(i) for i in decollate_batch(DX, detach=False)]
            y_pred_act = [self.post_pred(i) for i in decollate_batch(logits)]
            y_pred_act_onehot = [self.post_pred_argmax(i) for i in decollate_batch(logits)]
            self.cls_metrics(y_pred=y_pred_act_onehot, y=DX_onehot)
            self.AUC(y_pred=y_pred_act, y=DX_onehot)
        # logging metrics
        loss_CLS_results = self.loss_CLS_epoch.aggregate()
        loss_Q_results = self.loss_Q_epoch.aggregate()
        loss_Matching_results = self.loss_Matching_epoch.aggregate()
        cm_result = self.cls_metrics.aggregate()
        AUC_results = self.AUC.aggregate()
        if status == 'train':
            elapsed = time.time() - self.start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
            self.logger.print_message(f'Epoch {epoch + 1}/{args.epochs} - Elapsed time {elapsed}')
        self.logger.print_message(f'{status}    - Loss_CLS:{float(loss_CLS_results):.4f} '
                                  f'Loss_Q:{float(loss_Q_results):.4f} '
                                  f'Loss_Matching:{float(loss_Matching_results):.4f} '
                                  f'ACC:{float(cm_result[0]):.4f} '
                                  f'SEN:{float(cm_result[1]):.4f} '
                                  f'SPE:{float(cm_result[2]):.4f} '
                                  f'F1:{float(cm_result[3]):.4f} '
                                  f'AUC:{AUC_results:.4f}')
        # save best model according to the validation results
        if status == 'eval':
            if epoch >= args.warmup_epochs:
                acc = float(cm_result[0].cpu())
                f1 = float(cm_result[3].cpu())
                if acc >= self.best_acc and f1 >= self.best_f1:
                    self._save_checkpoint()
                    self.best_acc = acc
                    self.best_f1 = f1
                    self.best_epoch = epoch + 1
        # reset metrics
        self.loss_CLS_epoch.reset()
        self.loss_Q_epoch.reset()
        self.loss_Matching_epoch.reset()
        self.cls_metrics.reset()
        self.AUC.reset()
        metric_res = [float(loss_CLS_results), float(loss_Q_results), float(cm_result[0]),
                      float(cm_result[1]), float(cm_result[2]), float(cm_result[3]), AUC_results]
        return metric_res

    def start_train(self, train_loader, val_loader, test_loader):
        args = self.args
        # use acc & f1 score to select model
        self.best_acc = 0.0
        self.best_f1 = 0.0
        self.best_epoch = 1
        self.logger.print_message('Start training...')
        self.start_time = time.time()
        # start training
        for epoch in range(0, args.epochs):
            # training iterations
            _ = self.one_epoch(train_loader, 'train', epoch, args)
            # validation iterations
            with torch.no_grad():
                _ = self.one_epoch(val_loader, 'eval', epoch, args)
        # start test iterations
        #   load best model
        self.logger.print_message(f'\nLoad best model at epoch {self.best_epoch:03d}')
        self._load_checkpoint()
        #   test iterations
        with torch.no_grad():
            test_res = self.one_epoch(test_loader, 'test', self.best_epoch, args)
        return test_res

    def start_test(self, test_loader):
        args = self.args
        # start test iterations
        #   load best model
        self._load_checkpoint()
        #   test epoch
        with torch.no_grad():
            test_res = self.one_epoch(test_loader, 'test', args.epochs, args)
        return test_res


class model_CLS_VQCNN_MULTIMODEL_Transformer(nn.Module):
    def __init__(self, args, fold):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = build_models(args, 'MULTIMODEL_Transformer')
        # create checkpoint dir
        self.fold = fold
        self.checkpoint_dir = os.path.join(args.checkpoints_dir, args.name, str(fold))
        if not os.path.exists(self.checkpoint_dir):
            utils.mkdir(self.checkpoint_dir)
        self.logger = utils.Logger(self.checkpoint_dir)
        # print network status
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        # create optimizers and checkpoints
        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                if net == 'Trans' or net == 'CLS':
                    self.optims[net] = torch.optim.AdamW(
                        params=self.nets[net].parameters(), lr=args.lr, betas=(0.9, 0.999),
                        eps=1e-08, weight_decay=args.weight_decay)
        Trans_net = Munch(Trans=self.nets['Trans'])
        CLS_net = Munch(CLS=self.nets['CLS'])
        self.ckptios = {'Trans': utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_Trans_nets.ckpt'), **Trans_net),
                        'CLS': utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_CLS_nets.ckpt'), **CLS_net)}
        # to CUDA device
        self.to(self.device)
        # transform utils
        self.post_pred = Compose([Activations(softmax=True)])
        self.post_pred_argmax = Compose([AsDiscrete(argmax=True, to_onehot=args.code_num)])
        self.post_label = Compose([AsDiscrete(to_onehot=args.code_num)])
        self.post_pred_argmax_CLS = Compose([AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label_CLS = Compose([AsDiscrete(to_onehot=2)])
        # define evaluation metrics
        self.loss_epoch = CumulativeAverage()
        self.cls_metrics = ConfusionMatrixMetric(metric_name=["accuracy", 'sensitivity', 'specificity', 'f1 score'],
                                                 include_background=False, reduction='mean')
        self.AUC = ROCAUCMetric(average='micro')

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def _save_checkpoint(self, name):
        self.ckptios[name].save()

    def _load_checkpoint(self, name):
        self.ckptios[name].load()

    def _load_pretrain_nets(self):
        fname_MRI = os.path.join(self.checkpoint_dir, 'best_MRI_nets.ckpt')
        fname_PET = os.path.join(self.checkpoint_dir, 'best_PET_nets.ckpt')
        assert os.path.exists(fname_MRI), fname_MRI + ' does not exist!'
        assert os.path.exists(fname_PET), fname_PET + ' does not exist!'

        print('Loading MRI checkpoint from %s...' % fname_MRI)
        print('Loading PET checkpoint from %s...' % fname_PET)

        if torch.cuda.is_available():
            MRI_module_dict = torch.load(fname_MRI)
            PET_module_dict = torch.load(fname_PET)
        else:
            MRI_module_dict = torch.load(fname_MRI, map_location=torch.device('cpu'))
            PET_module_dict = torch.load(fname_PET, map_location=torch.device('cpu'))

        for name, module in self.nets.items():
            if name == 'MRICNN':
                module.load_state_dict(MRI_module_dict['CNN'])
            elif name == 'MRICODEBOOK':
                module.load_state_dict(MRI_module_dict['CODEBOOK'])
            elif name == 'MRICLS':
                module.load_state_dict(MRI_module_dict['CLS'])
            elif name == 'PETCNN':
                module.load_state_dict(PET_module_dict['CNN'])
            elif name == 'PETCODEBOOK':
                module.load_state_dict(PET_module_dict['CODEBOOK'])
            elif name == 'PETCLS':
                module.load_state_dict(PET_module_dict['CLS'])
            else:
                print(f'Skip loading {name}')
                continue

    def compute_loss(self, MRI, FDG):
        MRI_feats = self.nets.MRICNN(MRI)
        PET_feats = self.nets.PETCNN(FDG)
        # quant feats
        MRI_feats = rearrange(MRI_feats, 'b c h w d -> b (h w d) c')
        PET_feats = rearrange(PET_feats, 'b c h w d -> b (h w d) c')
        MRI_feats_quant, MRI_indices, _ = self.nets.MRICODEBOOK(MRI_feats) # (b*n)
        _, PET_indices, _ = self.nets.PETCODEBOOK(PET_feats) # (b*n)
        MRI_indices = MRI_indices.view(MRI.shape[0], -1) # (b, n)
        PET_indices = PET_indices.view(FDG.shape[0], -1) # (b, n)
        logits, _ = self.nets.Trans(MRI_indices) # (b, n, num_codebook)

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), PET_indices.reshape(-1))
        return logits, MRI_feats_quant, loss

    def predict_PET(self, MRI):
        MRI_feats = self.nets.MRICNN(MRI)
        MRI_feats = rearrange(MRI_feats, 'b c h w d -> b (h w d) c')
        MRI_feats_quant, MRI_indices, _ = self.nets.MRICODEBOOK(MRI_feats) # (b*n)
        MRI_indices = MRI_indices.view(MRI.shape[0], -1) # (b, n)
        logits, _ = self.nets.Trans(MRI_indices) # (b, n, num_codebook)
        return logits, MRI_feats_quant

    def compute_CLS_loss(self, MRI_feats, PET_logits, DX, stage):
        PET_feats = self.nets.PETCODEBOOK.get_vec_from_logits(PET_logits)
        PET_feats = rearrange(PET_feats, 'b n c -> b c n')
        MRI_feats = rearrange(MRI_feats, 'b n c -> b c n')
        PET_feats = F.adaptive_max_pool1d(PET_feats, 1).view(PET_feats.shape[0], -1)
        MRI_feats = F.adaptive_max_pool1d(MRI_feats, 1).view(MRI_feats.shape[0], -1)
        if stage == 2:
            logits = self.nets.PETCLS(PET_feats)
        else:
            logits = self.nets.CLS(torch.cat([MRI_feats, PET_feats], dim=1))
        loss_CLS = F.cross_entropy(logits, DX)
        return logits, loss_CLS

    def train_epoch(self, loader, epoch, args, stage):
        for name, model in self.nets.items():
            if stage == 2:
                if name == 'Trans':
                    unfreeze_model(model)
                    model.train()
                else:
                    freeze_model(model)
                    model.eval()
            else:
                if name == 'CLS':
                    unfreeze_model(model)
                    model.train()
                else:
                    freeze_model(model)
                    model.eval()
        # start epoch
        for i, data in enumerate(loader):
            # fetch images and labels
            MRI, FDG, _, _, DX = data
            MRI, FDG, DX = MRI.to(self.device), FDG.to(self.device), DX.to(self.device)
            pred_logits, MRI_feats, loss = self.compute_loss(MRI, FDG) # (b, n, code_num)
            logits, loss_CLS = self.compute_CLS_loss(MRI_feats, pred_logits, DX, stage)
            self._reset_grad()
            if stage == 2:
                loss_all = loss + 0.5 * loss_CLS
                loss_all.backward()
                self.optims.Trans.step()
            else:
                loss_CLS.backward()
                self.optims.CLS.step()

            #    epoch average loss
            self.loss_epoch.append(loss)
            #    cls metrics
            DX_onehot = [self.post_label_CLS(i) for i in decollate_batch(DX, detach=False)]
            y_pred_act = [self.post_pred(i) for i in decollate_batch(logits)]
            y_pred_act_onehot = [self.post_pred_argmax_CLS(i) for i in decollate_batch(logits)]
            self.cls_metrics(y_pred=y_pred_act_onehot, y=DX_onehot)
            self.AUC(y_pred=y_pred_act, y=DX_onehot)
        # get metrics every epoch
        loss_results = self.loss_epoch.aggregate()
        cm_result = self.cls_metrics.aggregate()
        AUC_results = self.AUC.aggregate()
        # logging metrics
        elapsed = time.time() - self.start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
        if stage == 2:
            self.logger.print_message(f'Epoch {epoch + 1}/{args.epochs} - Elapsed time {elapsed}')
        else:
            self.logger.print_message(f'Epoch {epoch + 1}/{args.CLS_epochs} - Elapsed time {elapsed}')
        self.logger.print_message(f'Train      - Loss:{float(loss_results):.4f} '
                                  f'ACC:{float(cm_result[0]):.3f} SEN:{float(cm_result[1]):.3f} '
                                  f'SPE:{float(cm_result[2]):.3f} F1:{float(cm_result[3]):.3f} '
                                  f'AUC:{float(AUC_results):.3f}')
        # save best model according to the validation results
        # reset metrics
        self.loss_epoch.reset()
        self.cls_metrics.reset()
        self.AUC.reset()

    def eval_epoch(self, loader, status, stage):
        for name, model in self.nets.items():
            model.eval()
        # start epoch
        for i, data in enumerate(loader):
            # fetch images and labels
            MRI, _, _, _, DX = data
            MRI, DX = MRI.to(self.device), DX.to(self.device)
            pred_logits, MRI_feats = self.predict_PET(MRI) # (b, n, code_num)
            logits, loss_CLS = self.compute_CLS_loss(MRI_feats, pred_logits, DX, stage)
            #    epoch average loss
            self.loss_epoch.append(loss_CLS)
            #    cls metrics
            # DX_onehot = [self.post_label(i) for i in decollate_batch(targets.reshape(-1), detach=False)]
            # y_pred_act = [self.post_pred(i) for i in decollate_batch(logits.reshape(-1, logits.size(-1)))]
            # y_pred_act_onehot = [self.post_pred_argmax(i) for i in decollate_batch(logits.reshape(-1, logits.size(-1)))]
            DX_onehot = [self.post_label_CLS(i) for i in decollate_batch(DX, detach=False)]
            y_pred_act = [self.post_pred(i) for i in decollate_batch(logits)]
            y_pred_act_onehot = [self.post_pred_argmax_CLS(i) for i in decollate_batch(logits)]
            self.cls_metrics(y_pred=y_pred_act_onehot, y=DX_onehot)
            self.AUC(y_pred=y_pred_act, y=DX_onehot)
        # get metrics every epoch
        loss_results = self.loss_epoch.aggregate()
        cm_result = self.cls_metrics.aggregate()
        AUC_results = self.AUC.aggregate()
        # logging metrics
        self.logger.print_message(f'{status}       - Loss_CLS:{float(loss_results):.4f} '
                                  f'ACC:{float(cm_result[0]):.3f} SEN:{float(cm_result[1]):.3f} '
                                  f'SPE:{float(cm_result[2]):.3f} F1:{float(cm_result[3]):.3f} '
                                  f'AUC:{float(AUC_results):.3f}')
        # save best model according to the validation results
        if status == 'eval':
            acc = float(cm_result[0].cpu())
            auc = float(AUC_results)
            if auc >= self.best_auc:
                if stage == 2:
                    self._save_checkpoint('Trans')
                else:
                    self._save_checkpoint('CLS')
                self.best_acc = acc
                self.best_auc = auc
        # reset metrics
        self.loss_epoch.reset()
        self.cls_metrics.reset()
        self.AUC.reset()
        metric_res = [float(loss_results), float(cm_result[0]), float(cm_result[1]),
                      float(cm_result[2]), float(cm_result[3]), AUC_results]
        return metric_res

    def start_train(self, train_loader, val_loader):
        args = self.args
        self._load_pretrain_nets()
        # use acc & f1 score to select model
        self.best_acc = 0.0
        self.best_auc = 0.0
        self.logger.print_message('Start training...')
        self.start_time = time.time()
        # start training
        # stage 2
        self.logger.print_message('Training Stage 2...')
        for epoch in range(0, args.epochs):
            # training iterations
            self.train_epoch(train_loader, epoch, args, stage=2)
            # validation iterations
            with torch.no_grad():
                _ = self.eval_epoch(val_loader, 'eval', stage=2)
        self.logger.print_message('Training Stage 3... Loading Transformer')
        # stage 3
        self._load_checkpoint('Trans')
        self.best_acc = 0.0
        self.best_auc = 0.0
        for epoch in range(0, args.CLS_epochs):
            # training iterations
            self.train_epoch(train_loader, epoch, args, stage=3)
            # validation iterations
            with torch.no_grad():
                _ = self.eval_epoch(val_loader, 'eval', stage=3)
        # test iterations

    def start_test(self, test_loader, stage):
        self.logger.print_message(f'Training complete. Start test')
        self._load_pretrain_nets()
        self._load_checkpoint('Trans')
        self._load_checkpoint('CLS')
        with torch.no_grad():
            test_res = self.eval_epoch(test_loader, 'test', stage=stage)
        return test_res


class model_CLS_CNN_MULTIMODEL_Transformer(nn.Module):
    def __init__(self, args, fold):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = build_models(args, 'MULTIMODEL_Transformer_withoutVQ')
        # create checkpoint dir
        self.fold = fold
        self.checkpoint_dir = os.path.join(args.checkpoints_dir, args.name, str(fold))
        if not os.path.exists(self.checkpoint_dir):
            utils.mkdir(self.checkpoint_dir)
        self.logger = utils.Logger(self.checkpoint_dir)
        # print network status
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        # create optimizers and checkpoints
        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                if net == 'Trans' or net == 'CLS':
                    self.optims[net] = torch.optim.AdamW(
                        params=self.nets[net].parameters(), lr=args.lr, betas=(0.9, 0.999),
                        eps=1e-08, weight_decay=args.weight_decay)
        Trans_net = Munch(Trans=self.nets['Trans'])
        CLS_net = Munch(CLS=self.nets['CLS'])
        self.ckptios = {'Trans': utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_Trans_nets.ckpt'), **Trans_net),
                        'CLS': utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_CLS_nets.ckpt'), **CLS_net)}
        # to CUDA device
        self.to(self.device)
        # transform utils
        self.post_pred = Compose([Activations(softmax=True)])
        self.post_pred_argmax = Compose([AsDiscrete(argmax=True, to_onehot=args.code_num)])
        self.post_label = Compose([AsDiscrete(to_onehot=args.code_num)])
        self.post_pred_argmax_CLS = Compose([AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label_CLS = Compose([AsDiscrete(to_onehot=2)])
        # define evaluation metrics
        self.loss_epoch = CumulativeAverage()
        self.cls_metrics = ConfusionMatrixMetric(metric_name=["accuracy", 'sensitivity', 'specificity', 'f1 score'],
                                                 include_background=False, reduction='mean')
        self.AUC = ROCAUCMetric(average='micro')

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def _save_checkpoint(self, name):
        self.ckptios[name].save()

    def _load_checkpoint(self, name):
        self.ckptios[name].load()

    def _load_pretrain_nets(self):
        fname_MRI = os.path.join(self.checkpoint_dir, 'best_MRI_nets.ckpt')
        fname_PET = os.path.join(self.checkpoint_dir, 'best_PET_nets.ckpt')
        assert os.path.exists(fname_MRI), fname_MRI + ' does not exist!'
        assert os.path.exists(fname_PET), fname_PET + ' does not exist!'

        print('Loading MRI checkpoint from %s...' % fname_MRI)
        print('Loading PET checkpoint from %s...' % fname_PET)

        if torch.cuda.is_available():
            MRI_module_dict = torch.load(fname_MRI)
            PET_module_dict = torch.load(fname_PET)
        else:
            MRI_module_dict = torch.load(fname_MRI, map_location=torch.device('cpu'))
            PET_module_dict = torch.load(fname_PET, map_location=torch.device('cpu'))

        for name, module in self.nets.items():
            if name == 'MRICNN':
                module.load_state_dict(MRI_module_dict['CNN'])
            elif name == 'MRICLS':
                module.load_state_dict(MRI_module_dict['CLS'])
            elif name == 'PETCNN':
                module.load_state_dict(PET_module_dict['CNN'])
            elif name == 'PETCLS':
                module.load_state_dict(PET_module_dict['CLS'])
            else:
                print(f'Skip loading {name}')
                continue

    def compute_loss(self, MRI, FDG):
        MRI_feats = self.nets.MRICNN(MRI)
        PET_feats = self.nets.PETCNN(FDG)
        # quant feats
        MRI_feats = rearrange(MRI_feats, 'b c h w d -> b (h w d) c')
        PET_feats = rearrange(PET_feats, 'b c h w d -> b (h w d) c')
        PET_imputed_feats, _ = self.nets.Trans(MRI_feats)

        loss = F.mse_loss(PET_feats, PET_imputed_feats)
        return PET_imputed_feats, MRI_feats, loss

    def predict_PET(self, MRI):
        MRI_feats = self.nets.MRICNN(MRI)
        MRI_feats = rearrange(MRI_feats, 'b c h w d -> b (h w d) c')
        PET_imputed_feats, _ = self.nets.Trans(MRI_feats)

        return PET_imputed_feats, MRI_feats

    def compute_CLS_loss(self, MRI_feats, PET_feats, DX, stage):
        PET_feats = rearrange(PET_feats, 'b n c -> b c n')
        MRI_feats = rearrange(MRI_feats, 'b n c -> b c n')
        PET_feats = F.adaptive_max_pool1d(PET_feats, 1).view(PET_feats.shape[0], -1)
        MRI_feats = F.adaptive_max_pool1d(MRI_feats, 1).view(MRI_feats.shape[0], -1)
        if stage == 2:
            logits = self.nets.PETCLS(PET_feats)
        else:
            logits = self.nets.CLS(torch.cat([MRI_feats, PET_feats], dim=1))
        loss_CLS = F.cross_entropy(logits, DX)
        return logits, loss_CLS

    def train_epoch(self, loader, epoch, args, stage):
        for name, model in self.nets.items():
            if stage == 2:
                if name == 'Trans':
                    unfreeze_model(model)
                    model.train()
                else:
                    freeze_model(model)
                    model.eval()
            else:
                if name == 'CLS':
                    unfreeze_model(model)
                    model.train()
                else:
                    freeze_model(model)
                    model.eval()
        # start epoch
        for i, data in enumerate(loader):
            # fetch images and labels
            MRI, FDG, _, _, DX = data
            MRI, FDG, DX = MRI.to(self.device), FDG.to(self.device), DX.to(self.device)
            PET_pred_feats, MRI_feats, loss = self.compute_loss(MRI, FDG) # (b, n, code_num)
            logits, loss_CLS = self.compute_CLS_loss(MRI_feats, PET_pred_feats, DX, stage)
            self._reset_grad()
            if stage == 2:
                loss_all = loss + 0.5 * loss_CLS
                loss_all.backward()
                self.optims.Trans.step()
            else:
                loss_CLS.backward()
                self.optims.CLS.step()

            #    epoch average loss
            self.loss_epoch.append(loss)
            #    cls metrics
            DX_onehot = [self.post_label_CLS(i) for i in decollate_batch(DX, detach=False)]
            y_pred_act = [self.post_pred(i) for i in decollate_batch(logits)]
            y_pred_act_onehot = [self.post_pred_argmax_CLS(i) for i in decollate_batch(logits)]
            self.cls_metrics(y_pred=y_pred_act_onehot, y=DX_onehot)
            self.AUC(y_pred=y_pred_act, y=DX_onehot)
        # get metrics every epoch
        loss_results = self.loss_epoch.aggregate()
        cm_result = self.cls_metrics.aggregate()
        AUC_results = self.AUC.aggregate()
        # logging metrics
        elapsed = time.time() - self.start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
        self.logger.print_message(f'Epoch {epoch + 1}/{args.epochs} - Elapsed time {elapsed}')
        self.logger.print_message(f'Train      - Loss:{float(loss_results):.4f} '
                                  f'ACC:{float(cm_result[0]):.3f} SEN:{float(cm_result[1]):.3f} '
                                  f'SPE:{float(cm_result[2]):.3f} F1:{float(cm_result[3]):.3f} '
                                  f'AUC:{float(AUC_results):.3f}')
        # save best model according to the validation results
        # reset metrics
        self.loss_epoch.reset()
        self.cls_metrics.reset()
        self.AUC.reset()

    def eval_epoch(self, loader, status, stage):
        for name, model in self.nets.items():
            model.eval()
        # start epoch
        for i, data in enumerate(loader):
            # fetch images and labels
            MRI, _, _, _, DX = data
            MRI, DX = MRI.to(self.device), DX.to(self.device)
            PET_pred_feats, MRI_feats = self.predict_PET(MRI) # (b, n, code_num)
            logits, loss_CLS = self.compute_CLS_loss(MRI_feats, PET_pred_feats, DX, stage)
            #    epoch average loss
            self.loss_epoch.append(loss_CLS)
            #    cls metrics
            # DX_onehot = [self.post_label(i) for i in decollate_batch(targets.reshape(-1), detach=False)]
            # y_pred_act = [self.post_pred(i) for i in decollate_batch(logits.reshape(-1, logits.size(-1)))]
            # y_pred_act_onehot = [self.post_pred_argmax(i) for i in decollate_batch(logits.reshape(-1, logits.size(-1)))]
            DX_onehot = [self.post_label_CLS(i) for i in decollate_batch(DX, detach=False)]
            y_pred_act = [self.post_pred(i) for i in decollate_batch(logits)]
            y_pred_act_onehot = [self.post_pred_argmax_CLS(i) for i in decollate_batch(logits)]
            self.cls_metrics(y_pred=y_pred_act_onehot, y=DX_onehot)
            self.AUC(y_pred=y_pred_act, y=DX_onehot)
        # get metrics every epoch
        loss_results = self.loss_epoch.aggregate()
        cm_result = self.cls_metrics.aggregate()
        AUC_results = self.AUC.aggregate()
        # logging metrics
        self.logger.print_message(f'{status}       - Loss_CLS:{float(loss_results):.4f} '
                                  f'ACC:{float(cm_result[0]):.3f} SEN:{float(cm_result[1]):.3f} '
                                  f'SPE:{float(cm_result[2]):.3f} F1:{float(cm_result[3]):.3f} '
                                  f'AUC:{float(AUC_results):.3f}')
        # save best model according to the validation results
        if status == 'eval':
            acc = float(cm_result[0].cpu())
            auc = float(AUC_results)
            if auc >= self.best_auc:
                if stage == 2:
                    self._save_checkpoint('Trans')
                else:
                    self._save_checkpoint('CLS')
                self.best_acc = acc
                self.best_auc = auc
        # reset metrics
        self.loss_epoch.reset()
        self.cls_metrics.reset()
        self.AUC.reset()
        metric_res = [float(loss_results), float(cm_result[0]), float(cm_result[1]),
                      float(cm_result[2]), float(cm_result[3]), AUC_results]
        return metric_res

    def start_train(self, train_loader, val_loader):
        args = self.args
        self._load_pretrain_nets()
        # use acc & f1 score to select model
        self.best_acc = 0.0
        self.best_auc = 0.0
        self.logger.print_message('Start training...')
        self.start_time = time.time()
        # start training
        # stage 2
        self.logger.print_message('Training Stage 2...')
        for epoch in range(0, args.epochs):
            # training iterations
            self.train_epoch(train_loader, epoch, args, stage=2)
            # validation iterations
            with torch.no_grad():
                _ = self.eval_epoch(val_loader, 'eval', stage=2)
        self.logger.print_message('Training Stage 3... Loading Transformer')
        # stage 3
        self._load_checkpoint('Trans')
        self.best_acc = 0.0
        self.best_auc = 0.0
        for epoch in range(0, args.CLS_epochs):
            # training iterations
            self.train_epoch(train_loader, epoch, args, stage=3)
            # validation iterations
            with torch.no_grad():
                _ = self.eval_epoch(val_loader, 'eval', stage=3)
        # test iterations

    def start_test(self, test_loader, stage):
        self.logger.print_message(f'Training complete. Start test')
        self._load_pretrain_nets()
        self._load_checkpoint('Trans')
        self._load_checkpoint('CLS')
        with torch.no_grad():
            test_res = self.eval_epoch(test_loader, 'test', stage=stage)
        return test_res
