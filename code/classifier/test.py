#!/usr/bin/env python
# coding: utf-8
import sys
from vision_transformer_pytorch import VisionTransformer

from glob import glob
from sklearn.model_selection import GroupKFold, StratifiedKFold
import cv2
from skimage import io
import torch
from torch import nn
import os
from datetime import datetime
import time
import random
import cv2
import torchvision
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.cuda.amp import autocast, GradScaler

import sklearn
import warnings
import joblib
from sklearn.metrics import roc_auc_score, log_loss
from sklearn import metrics
import warnings
import cv2
import pydicom
import timm
from scipy.ndimage.interpolation import zoom
from sklearn.metrics import log_loss

CFG1 = {
    'fold_num': 0,
    'seed': 88,
    #'model_arch': 'tf_efficientnet_b5_ns',
    'model_arch': 'wide_resnet50_2',
    'img_size': 128,
    'valid_bs': 32,
    'num_workers': 4,
    'device': 'cuda:0',
    'tta': 5,
    'used_epochs': [526],   #which epoch to be used 524
    'weights': [1]
}

train = pd.read_csv('./data/train.csv')
#train.label.value_counts()

# > We could do stratified validation split in each fold to make each fold's train and validation set looks like the whole train set in target distributions.
# submission = pd.read_csv('./result/submission.csv')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb

# # Dataset
class Dataset(Dataset):
    def __init__(
        self, df, data_root, transforms=None, output_label=True
    ):

        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        
        # get labels
        if self.output_label:
            target = self.df.iloc[index]['label']
          
        path = "{}/{}".format(self.data_root, self.df.iloc[index]['image_id'])
        
        img = get_img(path)
        
        if self.transforms:
            img = self.transforms(image=img)['image']
            
        # do label smoothing
        if self.output_label == True:
            return img, target
        else:
            return img


# # Define Train\Validation Image Augmentations
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2

def get_valid_transforms():
    return Compose([
            #CenterCrop(CFG1['img_size'], CFG1['img_size'], p=1.),
            Resize(CFG1['img_size'], CFG1['img_size']),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

def get_inference_transforms():
    return Compose([
            RandomResizedCrop(CFG1['img_size'], CFG1['img_size']),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

def get_inference_transforms2():
    return Compose([
            RandomResizedCrop(CFG1['img_size'], CFG1['img_size']),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

# Model Def
class ImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        #n_features = self.model.classifier.in_features
        n_features = self.model.fc.in_features
        self.model.classifier = nn.Linear(n_features, n_class)
        
    def forward(self, x):
        x = self.model(x)
        return x

# Model Ensemble
class EnsembleClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        #self.model1 = VisionTransformer.from_name('ViT-B_16', num_classes=5)
        #self.model1.load_state_dict(torch.load('../input/vit-model-1/ViT-B_16.pt'))
        self.model2 = ImgClassifier(model_arch, n_class, pretrained)
        
    def forward(self, x):
        #x1 = self.model1(x)
        x2 = self.model2(x)
        return x2
    
    def load(self, state_dict):
        self.model2.load_state_dict(state_dict)


# # Main Loop
def inference_one_epoch(model, data_loader, device):
    model.eval()
    image_preds_all = []
    image_targets_all = []
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        
        image_preds = model(imgs)   #output = model(input)
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]

        image_targets_all += [image_labels.detach().cpu().numpy()]
    
    image_preds_all = np.concatenate(image_preds_all, axis=0)
    image_targets_all = np.concatenate(image_targets_all)

    print('test multi-class accuracy = {:.4f}'.format((image_preds_all == image_targets_all).mean()))
    return image_preds_all

if __name__ == '__main__':
    seed_everything(CFG1['seed'])
    print('Inference started:')

    test = pd.DataFrame()
    #test['image_id'] = list(os.listdir('./data/test_images1/'))
    # dataset test.csv path
    test_csv_path = r'./data/test2.csv'   #test csv
    test = pd.read_csv(test_csv_path)
    test_ds1 = Dataset(test, './data/test_images4/', transforms=get_valid_transforms(), output_label=True) # we have label test2是老gan

    #构建测试集的
    #for tf_efficientnet_b5_ns
    tst_loader1 = torch.utils.data.DataLoader(
        test_ds1,
        batch_size=CFG1['valid_bs'],
        num_workers=CFG1['num_workers'],
        shuffle=False,
        pin_memory=False,
    )

    device = torch.device(CFG1['device'])
    tst_preds = []
    
    model = ImgClassifier(CFG1['model_arch'], train.label.nunique(), False).to(device)

    for i, epoch in enumerate(CFG1['used_epochs']):
        # load model parameter
        model.load_state_dict(torch.load('./models/{}_fold_{}_{}'.format(CFG1['model_arch'], CFG1['fold_num'], CFG1['used_epochs'][i])))

        with torch.no_grad():
            tst_preds = inference_one_epoch(model, tst_loader1, device)
            #valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False)

    #tst_preds2 = np.mean(tst_preds2, axis=0) 

    del model
    
    #tst_preds += tst_preds2
    # tst_preds = (tst_preds*tst_preds2)**5


    #tst_preds
    test['pred'] = tst_preds
    test['T or F'] = (test['pred'] == test['label'])
    test.to_csv('submission.csv', index=False)

