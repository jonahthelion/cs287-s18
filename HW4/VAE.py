import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
from torch.autograd import Variable
import torch.nn.functional as F

from utils.preprocess import get_data, get_model

"""
python VAE.py -model "Simple" -hidden 100 -lr .0008
"""

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
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(.9, .999))

for epoch in range(args.epochs):
    for data_ix,datum in enumerate(train_loader):
        model.train()
        img, label = datum
        mu, sig = model.get_encoding(Variable(img).cuda())
        z = mu + sig * Variable(torch.normal(mean=0.0, std=torch.ones(mu.shape[0],1))).cuda()
        img_out = model.get_decoding(z)

        l_reconstruct = F.binary_cross_entropy_with_logits(img_out.unsqueeze(1), Variable(img).cuda())
        l_kl = torch.stack([ 1./2*(s.sum()+m.pow(2).sum()-s.shape[0]-s.prod().log()) for m,s in zip(mu, sig)]).mean()
        print (data_ix, l_reconstruct, l_kl)