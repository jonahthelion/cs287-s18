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
python GAN.py -model "SimpleGAN" -hidden 150 -lr .001 -epochs 250
python GAN.py -model "CNNGAN" -hidden 150 -lr .001 -epochs 250
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
        img_g = model.get_decoding(z)

        # train the discriminator
        optimizer_d.zero_grad()
        l_d = 0
        preds = model.get_discrim(Variable(img.view(-1,28*28)).cuda()).view(-1)
        assert False
        gt = Variable(torch.zeros(len(preds)).cuda()) + Variable(torch.Tensor(len(preds)).uniform_(0,.2).cuda())
        l_d += F.binary_cross_entropy_with_logits(preds, gt)/2.
        preds = model.get_discrim(img_g.view(-1, 28*28)).view(-1)
        gt = Variable(torch.ones(len(preds)).cuda()) - Variable(torch.Tensor(len(preds)).uniform_(0,.2).cuda())
        l_d += F.binary_cross_entropy_with_logits(preds, gt)/2.
        l_d.backward()
        optimizer_d.step()
        optimizer_d.zero_grad(); optimizer_g.zero_grad();

        # train the generator
        z = Variable(torch.zeros(len(img), args.hidden).normal_().cuda())
        img_g = model.get_decoding(z)
        preds = model.get_discrim(img_g.view(-1, 28*28)).view(-1)
        gt = Variable(torch.zeros(len(preds)).cuda())+ Variable(torch.Tensor(len(preds)).uniform_(0.2).cuda())
        l_g = F.binary_cross_entropy_with_logits(preds, gt)
        l_g.backward()
        optimizer_g.step()
        optimizer_d.zero_grad(); optimizer_g.zero_grad();

        if data_ix % 300 == 0:
            print(data_ix, l_d, l_g)
            model.eval()

            # get sample images
            sample_z = Variable(torch.zeros(16, args.hidden).normal_().cuda())
            sample_img = model.get_decoding(sample_z)

            vis_windows = gan_display(vis, vis_windows, epoch + data_ix/float(len(train_loader)), l_d.data.cpu()[0], l_g.data.cpu()[0], (sample_img/2.+1/2.).data.cpu().unsqueeze(1))

# ##### GRID
# model = torch.load('GAN.p')
# model.eval()
# z = Variable(torch.zeros(36, args.hidden).normal_().cuda())
# img_g = model.get_decoding(z).data.cpu()
# fig = plt.figure()
# gs = mpl.gridspec.GridSpec(6,6)
# current = 0
# for i in range(6):
#     for j in range(6):
#         ax = plt.subplot(gs[i,j])
#         plt.imshow(img_g[current]/2. + 1/2., cmap='Greys')
#         plt.axis('off')
#         current += 1
# plt.tight_layout()
# plt.savefig('gan_decode2.pdf')
# plt.close(fig)

# ####### INTERPOLATE
# model = torch.load('GAN.p')
# model.eval()
# z = Variable(torch.zeros(36, args.hidden).normal_().cuda())
# img_g = model.get_decoding(z).data.cpu()
# z = z.data.cpu()
# for start in range(20):
#     fig = plt.figure(figsize=(10,2))
#     gs = mpl.gridspec.GridSpec(1,12)
#     ax = plt.subplot(gs[0,0])
#     plt.imshow(img_g[start], cmap='Greys')
#     plt.axis('off')
#     ax = plt.subplot(gs[0,-1])
#     plt.imshow(img_g[start+1], cmap='Greys')
#     plt.axis('off')
#     diff = z[start+1]-z[start]
#     for push_ix,push in enumerate(np.linspace(0,1,10)):
#         ax = plt.subplot(gs[0,push_ix+1])
#         plt.imshow(model.get_decoding(Variable((z[start] + push_ix*diff).cuda()).unsqueeze(0))[0].data.cpu()/2.+1/2., cmap='Greys')
#         plt.axis('off')
#     plt.tight_layout()
#     plt.savefig('gan_' + str(start) + '.pdf')
#     plt.close(fig)


