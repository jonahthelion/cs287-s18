import torchtext
import torch
from torchtext.vocab import Vectors
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm
import numpy as np
import visdom

from utils.preprocess import get_data, get_model

# visdom
vis_windows = None
vis = visdom.Visdom()
vis.env = 'train'

#############################

model_dict = {'type': 'noAttention',
                'D': 200,
                'num_encode': 4}

train_iter, val_iter, DE, EN = get_data(model_dict)

model = get_model(model_dict, DE, EN)

for batch in train_iter:
    preds = model.train_predict(batch.src.cuda(), batch.trg.cuda())
    assert False