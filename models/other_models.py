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

class DeepGuidanceModel(nn.Module):
    def __init__(self, args, fold):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = build_models(args, 'DeepGuidance')
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
                if net == 'GUIDE' or net == 'CLS':
                    self.optims[net] = torch.optim.AdamW(
                        params=self.nets[net].parameters(), lr=args.lr, betas=(0.9, 0.999),
                        eps=1e-08, weight_decay=args.weight_decay)
        GUIDE_net = Munch(Trans=self.nets['GUIDE'])
        CLS_net = Munch(CLS=self.nets['CLS'])
        self.ckptios = {'GUIDE': utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_GUIDE_nets.ckpt'), **GUIDE_net),
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

    # compute guidance loss
    def compute_loss(self, MRI, FDG):
        MRI_feats = self.nets.MRICNN(MRI)
        PET_feats = self.nets.PETCNN(FDG)
        b, c, h, w, d = MRI_feats.shape
        MRI_feats = rearrange(MRI_feats, 'b c h w d -> b (c h w d)')
        # map feats
        fake_PET_feats = self.nets.GUIDE(MRI_feats)
        fake_PET_feats = fake_PET_feats.view(b, c, h, w, d)
        MRI_feats = MRI_feats.view(b, c, h, w, d)
        # calculate loss
        loss = F.mse_loss(fake_PET_feats, PET_feats)
        return MRI_feats, fake_PET_feats, loss

    # predict PET features according to MRI
    def predict_PET(self, MRI):
        MRI_feats = self.nets.MRICNN(MRI)
        b, c, h, w, d = MRI_feats.shape
        MRI_feats = rearrange(MRI_feats, 'b c h w d -> b (c h w d)')
        fake_PET_feats = self.nets.GUIDE(MRI_feats)
        fake_PET_feats = fake_PET_feats.view(b, c, h, w, d)
        MRI_feats = MRI_feats.view(b, c, h, w, d)
        return MRI_feats, fake_PET_feats

    # predict label
    def compute_CLS_loss(self, MRI_feats, PET_feats, DX, stage):
        MRI_feats = rearrange(MRI_feats, 'b c h w d -> b c (h w d)')
        PET_feats = rearrange(PET_feats, 'b c h w d -> b c (h w d)')

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
                if name == 'GUIDE':
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
            MRI_feats, fake_PET_feats, loss = self.compute_loss(MRI, FDG) # (b, n, code_num)
            logits, loss_CLS = self.compute_CLS_loss(MRI_feats, fake_PET_feats, DX, stage)
            self._reset_grad()
            if stage == 2:
                loss_all = loss + loss_CLS
                loss_all.backward()
                self.optims.GUIDE.step()
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
            MRI_feats, fake_PET_feats = self.predict_PET(MRI) # (b, n, code_num)
            logits, loss_CLS = self.compute_CLS_loss(MRI_feats, fake_PET_feats, DX, stage)
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
                    self._save_checkpoint('GUIDE')
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
        self.logger.print_message('Training Stage 3... Loading Guidance Model')
        # stage 3
        self._load_checkpoint('GUIDE')
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
        self._load_checkpoint('GUIDE')
        self._load_checkpoint('CLS')
        with torch.no_grad():
            test_res = self.eval_epoch(test_loader, 'test', stage=stage)
        return test_res


class pix2pixelGANModel(nn.Module):
    def __init__(self, args, fold):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = build_models(args, 'pix2pixGAN')
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
        self.optims = Munch()
        for net, model in self.nets.items():
            if net == 'GAN':
                freeze_model(model)
                model.eval()
                continue
            self.optims[net] = torch.optim.AdamW(
                params=self.nets[net].parameters(), lr=args.lr, betas=(0.9, 0.999),
                eps=1e-08, weight_decay=args.weight_decay)
        nets = Munch(MRICNN=self.nets['MRICNN'], PETCNN=self.nets['PETCNN'], CLS=self.nets['CLS'])
        self.ckptios = [
            utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **nets)]
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

    def load_pretrained_GAN(self):
        print("=> Loading checkpoint")
        checkpoints_file = os.path.join(self.checkpoint_dir, 'best_GAN_nets.pth.tar')
        checkpoint = torch.load(checkpoints_file, map_location=self.device)
        self.nets.GAN.load_state_dict(checkpoint["state_dict"])

    def compute_CLS_loss(self, MRI, label):
        # impute PET images
        with torch.no_grad():
            PET = self.nets.GAN(MRI)
        # forward CNN
        MRI_feats = self.nets.MRICNN(MRI)
        PET_feats = self.nets.PETCNN(PET)
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
                if name == 'GAN':
                    continue
                model.train()
            for i, data in enumerate(train_loaders):
                # fetch images and labels
                MRI, _, _, _, DX = data
                MRI, DX = MRI.to(self.device), DX.to(self.device)
                logits, loss_CLS = self.compute_CLS_loss(MRI, DX)
                # backward
                self._reset_grad()
                loss_CLS.backward()
                self.optims.MRICNN.step()
                self.optims.PETCNN.step()
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
                    MRI, _, _, _, DX = data
                    MRI, DX = MRI.to(self.device),  DX.to(self.device)
                    logits, loss_CLS = self.compute_CLS_loss(MRI, DX)
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
                MRI, _, _, _, DX = data
                MRI, DX = MRI.to(self.device), DX.to(self.device)
                logits, loss_CLS = self.compute_CLS_loss(MRI, DX)
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


class pix2pixelGANModel_Single(nn.Module):
    def __init__(self, args, fold):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = build_models(args, 'pix2pixGAN_Single')
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
        self.optims = Munch()
        for net, model in self.nets.items():
            if net == 'GAN':
                freeze_model(model)
                model.eval()
                continue
            self.optims[net] = torch.optim.AdamW(
                params=self.nets[net].parameters(), lr=args.lr, betas=(0.9, 0.999),
                eps=1e-08, weight_decay=args.weight_decay)
        nets = Munch(PETCNN=self.nets['PETCNN'], CLS=self.nets['CLS'])
        self.ckptios = [
            utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **nets)]
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

    def load_pretrained_GAN(self):
        print("=> Loading checkpoint")
        checkpoints_file = os.path.join(self.checkpoint_dir, 'best_GAN_nets.pth.tar')
        checkpoint = torch.load(checkpoints_file, map_location=self.device)
        self.nets.GAN.load_state_dict(checkpoint["state_dict"])

    def compute_CLS_loss(self, MRI, label):
        # impute PET images
        with torch.no_grad():
            PET = self.nets.GAN(MRI)
        # forward CNN
        PET_feats = self.nets.PETCNN(PET)
        # Average Pool
        PET_feats = F.adaptive_max_pool3d(PET_feats, 1).view(PET.shape[0], -1)
        # forward MLP
        logits = self.nets.CLS(PET_feats)
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
                if name == 'GAN':
                    continue
                model.train()
            for i, data in enumerate(train_loaders):
                # fetch images and labels
                MRI, _, _, _, DX = data
                MRI, DX = MRI.to(self.device), DX.to(self.device)
                logits, loss_CLS = self.compute_CLS_loss(MRI, DX)
                # backward
                self._reset_grad()
                loss_CLS.backward()
                self.optims.PETCNN.step()
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
                    MRI, _, _, _, DX = data
                    MRI, DX = MRI.to(self.device),  DX.to(self.device)
                    logits, loss_CLS = self.compute_CLS_loss(MRI, DX)
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
                MRI, _, _, _, DX = data
                MRI, DX = MRI.to(self.device), DX.to(self.device)
                logits, loss_CLS = self.compute_CLS_loss(MRI, DX)
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


class FGANModel(nn.Module):
    def __init__(self, args, fold):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = build_models(args, 'FGAN')
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
        self.optims = Munch()
        for net, model in self.nets.items():
            if net == 'GAN':
                freeze_model(model)
                model.eval()
                continue
            self.optims[net] = torch.optim.AdamW(
                params=self.nets[net].parameters(), lr=args.lr, betas=(0.9, 0.999),
                eps=1e-08, weight_decay=args.weight_decay)
        nets = Munch(MRICNN=self.nets['MRICNN'], PETCNN=self.nets['PETCNN'], CLS=self.nets['CLS'])
        self.ckptios = [
            utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **nets)]
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

    def load_pretrained_GAN(self):
        print("=> Loading checkpoint")
        checkpoints_file = os.path.join(self.checkpoint_dir, 'gen.pth.tar')
        checkpoint = torch.load(checkpoints_file, map_location=self.device)
        self.nets.GAN.load_state_dict(checkpoint["state_dict"])

    def compute_CLS_loss(self, MRI, label):
        # impute PET images
        with torch.no_grad():
            PET = self.nets.GAN(MRI)
        # forward CNN
        MRI_feats = self.nets.MRICNN(MRI)
        PET_feats = self.nets.PETCNN(PET)
        # # Average Pool
        # MRI_feats = F.adaptive_max_pool3d(MRI_feats, 1).view(MRI.shape[0], -1)
        # PET_feats = F.adaptive_max_pool3d(PET_feats, 1).view(PET.shape[0], -1)
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
                if name == 'GAN':
                    continue
                model.train()
            for i, data in enumerate(train_loaders):
                # fetch images and labels
                MRI, _, _, _, DX = data
                MRI, DX = MRI.to(self.device), DX.to(self.device)
                logits, loss_CLS = self.compute_CLS_loss(MRI, DX)
                # backward
                self._reset_grad()
                loss_CLS.backward()
                self.optims.MRICNN.step()
                self.optims.PETCNN.step()
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
                    MRI, _, _, _, DX = data
                    MRI, DX = MRI.to(self.device),  DX.to(self.device)
                    logits, loss_CLS = self.compute_CLS_loss(MRI, DX)
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
                MRI, _, _, _, DX = data
                MRI, DX = MRI.to(self.device), DX.to(self.device)
                logits, loss_CLS = self.compute_CLS_loss(MRI, DX)
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


class FGANModel_Single(nn.Module):
    def __init__(self, args, fold):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets = build_models(args, 'FGAN_Single')
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
        self.optims = Munch()
        for net, model in self.nets.items():
            if net == 'GAN':
                freeze_model(model)
                model.eval()
                continue
            self.optims[net] = torch.optim.AdamW(
                params=self.nets[net].parameters(), lr=args.lr, betas=(0.9, 0.999),
                eps=1e-08, weight_decay=args.weight_decay)
        nets = Munch(PETCNN=self.nets['PETCNN'], CLS=self.nets['CLS'])
        self.ckptios = [
            utils.CheckpointIO(os.path.join(self.checkpoint_dir, 'best_nets.ckpt'), **nets)]
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

    def load_pretrained_GAN(self):
        print("=> Loading checkpoint")
        checkpoints_file = os.path.join(self.checkpoint_dir, 'gen.pth.tar')
        checkpoint = torch.load(checkpoints_file, map_location=self.device)
        self.nets.GAN.load_state_dict(checkpoint["state_dict"])

    def compute_CLS_loss(self, MRI, label):
        # impute PET images
        with torch.no_grad():
            PET = self.nets.GAN(MRI)
        # forward CNN
        PET_feats = self.nets.PETCNN(PET)
        # # Average Pool
        # PET_feats = F.adaptive_max_pool3d(PET_feats, 1).view(PET.shape[0], -1)
        # forward MLP
        logits = self.nets.CLS(PET_feats)
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
                if name == 'GAN':
                    continue
                model.train()
            for i, data in enumerate(train_loaders):
                # fetch images and labels
                MRI, _, _, _, DX = data
                MRI, DX = MRI.to(self.device), DX.to(self.device)
                logits, loss_CLS = self.compute_CLS_loss(MRI, DX)
                # backward
                self._reset_grad()
                loss_CLS.backward()
                self.optims.PETCNN.step()
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
                    MRI, _, _, _, DX = data
                    MRI, DX = MRI.to(self.device),  DX.to(self.device)
                    logits, loss_CLS = self.compute_CLS_loss(MRI, DX)
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
                MRI, _, _, _, DX = data
                MRI, DX = MRI.to(self.device), DX.to(self.device)
                logits, loss_CLS = self.compute_CLS_loss(MRI, DX)
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

