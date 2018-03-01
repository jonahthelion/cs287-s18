import torch
import torch.nn.functional as F
import torchtext
from torch.autograd import Variable

import numpy as np
from tqdm import tqdm

def vis_display(vis, vis_windows, x_coord, train_l, MAP):
    if vis_windows is None:
        vis_windows = {}
        vis_windows['train_ce'] = vis.line(Y=torch.Tensor([float(2**train_l)]), X=torch.Tensor([x_coord]), opts=dict(title='Train PPL'))
        vis_windows['val_ce'] = vis.line(Y=torch.Tensor([float(2**MAP)]), X=torch.Tensor([x_coord]), opts=dict(title='Validation PPL'))
    else:
        vis.line(Y=torch.Tensor([float(2**train_l)]), X=torch.Tensor([x_coord]), win=vis_windows['train_ce'], update='append', opts=dict(title='Train PPL'))
        if not MAP is None:
            vis.line(Y=torch.Tensor([float(2**MAP)]), X=torch.Tensor([x_coord]), win=vis_windows['val_ce'], update='append', opts=dict(title='Validation PPL'))
    return vis_windows

def evaluate(model, val_iter):
    all_losses = []
    for batch in val_iter:
        if batch.src.shape[1] == 32:
            output, hidden = model.get_encode(batch.src.cuda())
            output, hidden = model.get_decode(batch.trg.cuda(), hidden)
            loss = F.cross_entropy(output[:-1].view(-1, output.shape[-1]), batch.trg.cuda()[1:].view(-1), ignore_index=1)
            all_losses.append(loss)
    loss_v = torch.cat(all_losses).mean().data.cpu().numpy()[0]

    return loss_v