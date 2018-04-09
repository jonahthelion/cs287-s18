import torch
import argparse
from torch.utils import data as torchdata

from utils.preprocess import get_data

"""
python3 train.py -d ../multi30k-dataset -batch_size 4
"""

def get_args():
    parser = argparse.ArgumentParser(description='Train unsupervised model.')
    parser.add_argument('-d', metavar='-d', type=str,
                        help='directory multi30k was cloned to (https://github.com/multi30k/dataset)')
    parser.add_argument('-batch_size', type=int,
                        help='batch_size for training and validating dataloaders')
    return parser.parse_args()

args = get_args()
trainloader, valloader, TEXT = get_data(args)
trainbatcher = torchdata.DataLoader(trainloader, batch_size=args.batch_size, num_workers=4, shuffle=True)
valbatcher = torchdata.DataLoader(valloader, batch_size=args.batch_size, num_workers=4, shuffle=False)

for batch_ix,batch in enumerate(trainbatcher):
    print(batch_ix, batch)
    print([TEXT.itos[i][1] for i in batch['en'][0] if not i==-2])
    print([TEXT.itos[i][1] for i in batch['de'][0] if not i==-2])
    print([TEXT.itos[i][1] for i in batch['fr'][0] if not i==-2])
    assert False