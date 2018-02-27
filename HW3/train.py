import torchtext
import torch
from torchtext.vocab import Vectors
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm
import numpy as np
import visdom

from utils.preprocess import get_data


# visdom
vis_windows = None
vis = visdom.Visdom()
vis.env = 'train'

#############################

model_dict = {}

train_iter, val_iter, DE, EN = get_data(model_dict)

for batch in train_iter:
    print(batch.src)
    print(batch.trg)
    assert False