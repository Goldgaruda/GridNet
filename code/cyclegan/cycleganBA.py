import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
from tqdm import tqdm

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt
from common_tools import set_seed, transform_invert

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=1, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=60, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="data", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation") # 8
parser.add_argument("--img_height", type=int, default=128, help="size of image height")  # 256
parser.add_argument("--img_width", type=int, default=128, help="size of image width")  # 256
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
opt = parser.parse_args()
# print(opt)

# Create sample and checkpoint directories
# os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
# os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()

if opt.epoch != 0:
    # Load pretrained models
    #G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth")
    G_BA.load_state_dict(torch.load("./GANmodel/G_AB_780.pth"))
else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
transforms_ = [
    # transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    # transforms.RandomCrop((opt.img_height, opt.img_width)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Test data loader
val_dataloader = DataLoader(
    ImageDataset("./data", transforms_=transforms_, unaligned=False, mode="test"),
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    num_workers=1,
)

# ----------
#  Generating satellite images
# ----------

if __name__ == '__main__':
    pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
    G_AB.eval()  # AB satellite to UAV
    # G_BA.eval()
    for step, (imgs) in pbar:

        """Saves a generated sample from the test set"""
        # real_A = Variable(imgs["A"].type(Tensor))
        # fake_B = G_AB(real_A)
        real_B = Variable(imgs["B"].type(Tensor))
        fake_A = G_BA(real_B)

        # real_B = make_grid(real_B, nrow=1, normalize=True)
        # fake_A = make_grid(fake_A, nrow=1, normalize=True)
        # real_B = transform_invert(real_B, transforms_)
        # plt.imshow(real_B)
        # plt.show()
        # real_B.save("./data/test_images2/%s.jpeg" % i)
        # Arange images along x-axis
        # real_A = make_grid(real_A, nrow=5, normalize=True)
        # real_B = make_grid(real_B, nrow=5, normalize=True)
        # fake_A = make_grid(fake_A, nrow=5, normalize=True)
        # fake_B = make_grid(fake_B, nrow=5, normalize=True)
        # Arange images along y-axis
        # image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
        Bname = ''.join(imgs["Bname"])
        #save_image(fake_A, "./data/test_260_GAN1/" + Bname, normalize=True)
        save_image(fake_A, "./data/test_130_GAN1/" + Bname, normalize=True)
        #save_image(real_B, "./data/test_images2/r_" + Bname, normalize=True)
        # from PIL import Image, ImageChops
        # import numpy as np

