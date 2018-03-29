import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
from torch.autograd import Variable

from utils.preprocess import get_data, get_model

"""
python VAE.py -model "Simple" -hidden 100
"""

def get_args():
    parser = argparse.ArgumentParser(description='Train/evaluate a VAE.')
    parser.add_argument('-model', metavar='-model', type=str,
                        help='type of model')
    parser.add_argument('-hidden', metavar='-hidden', type=int,
                        help='the size of z')

    return parser.parse_args()

args = get_args()
train_loader, val_loader, test_loader = get_data(args)
model = get_model(args)

for datum in train_loader:
    model.train()
    img, label = datum
    mu, sig = model.get_encoding(Variable(img).cuda())
    z = mu + sig * Variable(torch.normal(mean=0.0, std=torch.ones(mu.shape[0],1))).cuda()
    img_out = model.get_decoding(z)
    break