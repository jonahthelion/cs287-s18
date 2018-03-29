import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
from torch.autograd import Variable
import torch.nn.functional as F
import visdom
import numpy as np

from utils.preprocess import get_data, get_model
from utils.postprocess import vis_display

"""
python VAE.py -model "Simple" -hidden 20 -lr .0008 -epochs 10 -kl_lam 0.1
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
    parser.add_argument('-kl_lam', metavar='-kl_lam', type=float,
                        help='weight of the kl term')

    return parser.parse_args()

args = get_args()
train_loader, val_loader, test_loader = get_data(args)
model = get_model(args)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(.9, .999))

for epoch in range(args.epochs):
    for data_ix,datum in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()

        img, label = datum
        mu, sig = model.get_encoding(Variable(img).cuda())
        z = mu + sig * Variable(torch.normal(mean=0.0, std=torch.ones(mu.shape[0],1))).cuda()
        img_out = model.get_decoding(z)

        l_reconstruct = F.binary_cross_entropy_with_logits(img_out.unsqueeze(1), Variable(img).cuda())
        l_kl = torch.stack([ 1./2*(s.sum()+m.pow(2).sum()-s.shape[0]-s.prod().log()) for m,s in zip(mu, sig)]).mean()
        
        (l_reconstruct + args.kl_lam * l_kl).backward()
        optimizer.step()

        if data_ix%30 == 0:
            print (data_ix, l_reconstruct, l_kl)
            model.eval()

            # get sample images
            sample_z = Variable(torch.normal(mean=0.0, std=torch.ones(16,args.hidden))).cuda()
            sample_img = model.get_decoding(sample_z)
            
            val_reconstruct=None; val_kl=None;
            if data_ix % 150 == 0:
                # get validation loss
                val_reconstruct = []; val_kl = [];
                for datum in val_loader:
                    img, label = datum
                    mu, sig = model.get_encoding(Variable(img).cuda())
                    z = mu + sig * Variable(torch.normal(mean=0.0, std=torch.ones(mu.shape[0],1))).cuda()
                    img_out = model.get_decoding(z)

                    l_reconstruct = F.binary_cross_entropy_with_logits(img_out.unsqueeze(1), Variable(img).cuda())
                    l_kl = torch.stack([ 1./2*(s.sum()+m.pow(2).sum()-s.shape[0]-s.prod().log()) for m,s in zip(mu, sig)]).mean()
                    val_reconstruct.append(l_reconstruct.data.cpu()[0]); val_kl.append(l_kl.data.cpu()[0]); 
                val_kl = np.mean(val_kl); val_reconstruct = np.mean(val_reconstruct);
            vis_windows = vis_display(vis, vis_windows, epoch + data_ix/float(len(train_loader)), l_reconstruct.data.cpu()[0], l_kl.data.cpu()[0], F.sigmoid(sample_img).data.cpu().unsqueeze(1), val_kl, val_reconstruct)


