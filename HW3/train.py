import torchtext
import torch
from torchtext.vocab import Vectors
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm
import numpy as np
import visdom

from utils.preprocess import get_data, get_model
from utils.postprocess import vis_display, evaluate

# visdom
vis_windows = None
vis = visdom.Visdom()
vis.env = 'train'

#############################

model_dict = {'type': 'noAttention',
                'D': 200,
                'num_encode': 4,
                'num_decode': 4,
                'num_epochs': 10}

train_iter, val_iter, DE, EN = get_data(model_dict)

model = get_model(model_dict, DE, EN)

optimizer = torch.optim.Adam(model.parameters(), lr=.0001)

for epoch in range(model_dict['num_epochs']):
    for batch_num,batch in enumerate(train_iter):
        model.train()
        optimizer.zero_grad()
        preds = model.train_predict(batch.src.cuda(), batch.trg.cuda())
        loss = F.cross_entropy(preds[:-1].view(-1,preds.shape[-1]), batch.trg[1:].view(-1).cuda(), ignore_index=1)
        loss.backward()
        optimizer.step()

        if batch_num % 100 == 0:
            loss_t = loss.data.cpu().numpy()[0]
            loss_v = None
            if batch_num % 300 == 0:
                model.eval()
                loss_v = evaluate(model, val_iter)
                print(epoch, batch_num, loss_v)
            vis_windows = vis_display(vis, vis_windows, epoch + batch_num/float(len(train_iter)), loss_t, loss_v)