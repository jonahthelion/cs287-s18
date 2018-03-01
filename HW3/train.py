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
                'num_epochs': 50,
                'fake': True,
                'pickled_fields': True}

train_iter, val_iter, DE, EN = get_data(model_dict)

###########
model = torch.load('noAttention.p')
model.encoder.flatten_parameters()
model.decoder.flatten_parameters()
model.eval()
fname = 'PSET/source_test.txt'
with open(fname, 'rb') as reader:
    for line in reader:
        src = Variable(torch.Tensor([DE.vocab.stoi[s] for s in line.decode('utf-8').strip('\n').split(' ')]).long().unsqueeze(1))
        output, hidden = model.get_encode(src.cuda())

        poss_sentences = Variable(torch.Tensor([[2, 2]]).long().cuda())
        poss_scores = [0, 0]
        poss_hidden = (torch.stack([hidden[0][:,0] for _ in range(poss_sentences.shape[1])], 1), torch.stack([hidden[1][:,0] for _ in range(poss_sentences.shape[1])], 1))
        preds = model.get_decode(poss_sentences, poss_hidden)

        new_sentences = []; new_scores = [];
        for i in range(poss_sentences.shape[1]):
            best_pred_vals, best_pred_ixes = preds[0][-1, i].topk(5)
            for val_ix in range(best_pred_ixes):
                new_sentences.append(torch.cat(poss_sentences[:,i], best_pred_vals[val_ix]))
                new_scores.append(poss_scores[i] + best_pred_ixes[val_ix])

        assert False
###########


model = get_model(model_dict, DE, EN)

optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-4, betas=(.9, .999))

for epoch in range(model_dict['num_epochs']):
    for batch_num,batch in enumerate(train_iter):
        model.train()
        optimizer.zero_grad()

        output, hidden = model.get_encode(batch.src.cuda())
        output, hidden = model.get_decode(batch.trg.cuda(), hidden)
        loss = F.cross_entropy(output[:-1].view(-1, output.shape[-1]), batch.trg.cuda()[1:].view(-1), ignore_index=1)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5)
        optimizer.step()

        if batch_num % 100 == 0:
            loss_t = loss.data.cpu().numpy()[0]
            loss_v = None
            if batch_num % 300 == 0:
                model.eval()
                loss_v = evaluate(model, val_iter)
                print(epoch, batch_num, loss_v)
            vis_windows = vis_display(vis, vis_windows, epoch + batch_num/float(len(train_iter)), loss_t, loss_v)



