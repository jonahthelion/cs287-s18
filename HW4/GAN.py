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
from utils.postprocess import gan_display, get_validation_loss

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
train_loader, val_loader, test_loader = get_data(args, bern=False)
model = get_model(args)
optimizer_g = torch.optim.Adam(model.decoder.parameters(), lr=args.lr)
optimizer_d = torch.optim.Adam(model.discrim.parameters(), lr=args.lr)

for epoch in range(args.epochs):
    for data_ix,(img,label) in enumerate(train_loader):
        model.train()
        img = (2*(img - .5)).squeeze(1)
        z = Variable(torch.zeros(len(img), args.hidden).normal_().cuda())

        # train the discriminator
        optimizer_d.zero_grad()
        if np.random.choice([0,1]):
            preds = model.get_discrim(Variable(img.view(-1,28*28)).cuda()).view(-1)
            gt = Variable(torch.zeros(len(preds)).cuda())
            d_type = 'data'
        else:
            img_g = model.get_decoding(z).detach()
            preds = model.get_discrim(img_g.view(-1, 28*28)).view(-1)
            gt = Variable(torch.ones(len(preds)).cuda())
            d_type = 'gen'
        l_d = F.binary_cross_entropy_with_logits(preds, gt)
        l_d.backward()
        optimizer_d.step()

        # train the generator
        optimizer_g.zero_grad()
        img_g = model.get_decoding(z)
        preds = model.get_discrim(img_g.view(-1, 28*28)).view(-1)
        gt = Variable(torch.zeros(len(preds)).cuda())
        l_g = F.binary_cross_entropy_with_logits(preds, gt)
        l_g.backward()
        optimizer_g.step()

        if data_ix % 30 == 0:
            print(data_ix, l_d, l_g, d_type)
            model.eval()

            # get sample images
            sample_z = Variable(torch.normal(mean=0.0, std=torch.ones(16,args.hidden))).cuda()
            sample_img = model.get_decoding(sample_z)

            vis_windows = gan_display(vis, vis_windows, epoch + data_ix/float(len(train_loader)), l_d.data.cpu()[0], l_g.data.cpu()[0], (sample_img/2.+1/2.).data.cpu().unsqueeze(1))




