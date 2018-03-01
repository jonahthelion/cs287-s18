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
                'num_epochs': 10,
                'fake': True,
                'pickled_fields': True}

train_iter, val_iter, DE, EN = get_data(model_dict)

model = get_model(model_dict, DE, EN)

optimizer = torch.optim.Adam(model.parameters(), lr=.0001)

for epoch in range(model_dict['num_epochs']):
    for batch_num,batch in enumerate(train_iter):
        model.train()
        optimizer.zero_grad()

        output, hidden = model.get_encode(batch.src.cuda())
        output, hidden = model.get_decode(batch.trg.cuda(), hidden)
        loss = F.binary_cross_entropy_with_logits(output, Variable(torch.zeros(output.shape).cuda()))
        
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