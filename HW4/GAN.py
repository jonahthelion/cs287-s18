import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
from torch.autograd import Variable
import torch.nn.functional as F
import visdom
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from utils.preprocess import get_data, get_model
from utils.postprocess import vis_display, get_validation_loss

"""
python GAN.py -model "SimpleGAN" -hidden 2 -lr .001 -epochs 40
"""

# visdom
vis_windows = None
vis = visdom.Visdom()
vis.env = 'train'

def get_args():
    parser = argparse.ArgumentParser(description='Train/evaluate a VAE.')
    parser.add_argument('-model', metavar='-model', type=str,
                        help='type of model')
    parser.add_argument('-hidden', metavar='-hidden', type=int,
                        help='the size of z')
    parser.add_argument('-lr', metavar='-lr', type=float,
                        help='learning rate')
    parser.add_argument('-epochs', metavar='-epochs', type=int,
                        help='number of epochs')

    return parser.parse_args()

args = get_args()
train_loader, val_loader, test_loader = get_data(args)
model = get_model(args)
optimizer_g = torch.optim.Adam(model.decoder.parameters(), lr=args.lr)
optimizer_d = torch.optim.Adam(model.discrim.parameters(), lr=args.lr)

for epoch in range(args.epochs):
    for data_ix,(img,label) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        img = 2*(img - .5)

        assert False