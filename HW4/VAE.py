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
python VAE.py -model "Simple" -hidden 2 -lr .001 -epochs 40
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
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(args.epochs):
    for data_ix,(img,label) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()

        mu, logvar = model.get_encoding(Variable(img).cuda())
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        z = eps.mul(std).add_(mu)
        img_out = model.get_decoding(z)

        l_reconstruct = F.binary_cross_entropy_with_logits(img_out.view(-1,28*28), Variable(img).cuda().view(-1, 28*28), size_average=False) / float(len(mu))
        l_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / float(len(mu))
        
        (l_reconstruct + l_kl).backward()
        optimizer.step()

        if data_ix%30 == 0:
            print (data_ix, l_reconstruct, l_kl)
            model.eval()

            # get sample images
            sample_z = Variable(torch.normal(mean=0.0, std=torch.ones(16,args.hidden))).cuda()
            sample_img = model.get_decoding(sample_z)
            
            val_reconstruct=None; val_kl=None;
            if data_ix % 150 == 0:
                val_kl, val_reconstruct = get_validation_loss(model, val_loader)
            vis_windows = vis_display(vis, vis_windows, epoch + data_ix/float(len(train_loader)), l_reconstruct.data.cpu()[0], l_kl.data.cpu()[0], F.sigmoid(sample_img).data.cpu().unsqueeze(1), val_kl, val_reconstruct)

# model = torch.load('VAE.p')
# model.eval()
# all_mu = []; all_labels = [];
# for data in test_loader:
#     img, label = data
#     mu, sig = model.get_encoding(Variable(img).cuda())
#     all_mu.append(mu.data.cpu())
#     all_labels.append(label)
# all_mu = torch.cat(all_mu, 0); all_labels = torch.cat(all_labels, 0)
# fig = plt.figure()
# handles = []
# for dig in range(10):
#     ixes = [ix for ix in range(len(all_mu)) if all_labels[ix] == dig]
#     p = plt.plot([all_mu[ix,0] for ix in ixes], [all_mu[ix,1] for ix in ixes], '.')
#     handles.append(mpl.patches.Patch(color=p[0].get_color(), label=str(dig)))
# plt.legend(handles=handles)
# plt.tight_layout()
# plt.savefig('vae_mus2.pdf')



