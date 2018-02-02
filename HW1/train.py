import torch
import torchtext
from torchtext.vocab import Vectors, GloVe
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from sklearn import metrics
import visdom
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams.update({'font.size': 18})

from utils.preprocess import get_data
from models.psetModels import MNB, LogReg, CBOW, Conv
from utils.postprocess import print_important, vis_display, evaluate_model

# set up visdom
vis = visdom.Visdom()
vis.env = 'train'
vis_windows = None

# define model
# chosen_model = {'type': 'MNB', 'alpha':[.5], 'should_plot':False, 'counts': False}
chosen_model = {'type': 'log_reg'}
print(chosen_model)

# get data
TEXT, LABEL, train_iter, val_iter, test_iter = get_data(batch_size=50)


if chosen_model['type'] == 'MNB':
    all_scores = []
    for alpha in chosen_model['alpha']:
        model = MNB(V=len(TEXT.vocab), alpha=.1, counts=chosen_model['counts'])

        for batch_num,batch in enumerate(tqdm(train_iter)):
            model.train_sample(batch.label.data - 1, batch.text.data)
        model.postprocess()
        print_important(model.w.weight.data.cpu().squeeze(0), TEXT, 10)
        bce, roc, acc = evaluate_model(model, test_iter)
        print('alpha:', alpha)
        print('BCE:', bce, '  ROC:',roc,'  ACC:', acc)
        all_scores.append((bce, roc, acc))

    if chosen_model['should_plot']:
        fig = plt.figure(figsize=(10,6))
        plt.plot(chosen_model['alpha'], [all_score[2] for all_score in all_scores])
        plt.xlabel(r'$\alpha$')
        plt.ylabel('Accuracy')
        plt.savefig('writeup/imgs/alpha.pdf')
        plt.close(fig)


if chosen_model['type'] == 'log_reg':

    model = LogReg(V=len(TEXT.vocab))
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    for epoch in range(10):
        for batch_num,batch in enumerate(train_iter):
            optimizer.zero_grad()
            preds = model(batch.text.data)
            l = F.binary_cross_entropy_with_logits(preds.view(-1), Variable((batch.label - 1).float().cuda()))
            l.backward()
            optimizer.step()

            if batch_num % 40 == 0:
                vis_windows = vis_display(vis, vis_windows, l.cpu().data.numpy()[0], epoch + batch_num/float(len(train_iter)))
            if batch_num % 100 == 0:
                bce, roc, acc = evaluate_model(model, val_iter)
                vis_windows = vis_display(vis, vis_windows, l.cpu().data.numpy()[0], epoch + batch_num/float(len(train_iter)), bce)

    #         l = model.train_sample(batch.label.float() - 1, batch.text.data, optimizer)
    #         if batch_num % 100 != 0 and batch_num % 40 == 0:
    #             
    #         else:
    #             lvals = []
    #             for batch in val_iter:
    #                 lvals.append(model.evalu_loss(batch.label.float() - 1, batch.text.data).data.numpy()[0])
    #             vis_windows = vis_display(vis, vis_windows, l.data.numpy()[0], epoch + batch_num/float(len(train_iter)), sum(lvals)/float(len(lvals)))
    #     print('saving', str(epoch) + '_' + 'logreg.p')
    #     torch.save(model, str(epoch) + '_' + 'logreg.p')

    #     bad_vals,bad_ixes = torch.topk(model.w.weight.data, 10, largest=True)
    #     good_vals,good_ixes = torch.topk(model.w.weight.data, 10, largest=False)
    #     print_important(TEXT, bad_vals.squeeze(), bad_ixes.squeeze(), good_vals.squeeze(), good_ixes.squeeze())





if chosen_model['type'] == 'CBOW':
    model = torch.load('4_cbow.p')
    model = CBOW(V=len(TEXT.vocab), embed=TEXT.vocab.vectors)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    for epoch in range(5):
        model.train()
        for batch_num,batch in enumerate(train_iter):
            l = model.train_sample(batch.label.float() - 1, batch.text.data, optimizer)
            if batch_num % 100 != 0 and batch_num % 40 == 0:
                vis_windows = vis_display(vis, vis_windows, l.data.numpy()[0], epoch + batch_num/float(len(train_iter)))
            else:
                lvals = []
                for batch in val_iter:
                    lvals.append(model.evalu_loss(batch.label.float() - 1, batch.text.data).data.numpy()[0])
                vis_windows = vis_display(vis, vis_windows, l.data.numpy()[0], epoch + batch_num/float(len(train_iter)), sum(lvals)/float(len(lvals)))
        print('saving', str(epoch) + '_' + 'cbow.p')
        torch.save(model, str(epoch) + '_' + 'cbow.p')

        model.eval()
        model.evalu(test_iter)


if chosen_model['type'] == 'conv':
    model = Conv(V=len(TEXT.vocab), embed=TEXT.vocab.vectors)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    model.train()
    for epoch in range(5):
        model.train()
        for batch_num,batch in enumerate(train_iter):
            optimizer.zero_grad()
            outs = model(batch.text.data)
            l = F.binary_cross_entropy_with_logits(outs, batch.label.float() - 1)
            l.backward()
            optimizer.step()

            if batch_num % 100 != 0 and batch_num % 40 == 0:
                vis_windows = vis_display(vis, vis_windows, l.data.numpy()[0], epoch + batch_num/float(len(train_iter)))
            else:
                lvals = []
                for batch in val_iter:
                    lvals.append(model.evalu_loss(batch.label.float() - 1, batch.text.data).data.numpy()[0])
                vis_windows = vis_display(vis, vis_windows, l.data.numpy()[0], epoch + batch_num/float(len(train_iter)), sum(lvals)/float(len(lvals)))
        print('saving', str(epoch) + '_' + 'conv.p')
        torch.save(model, str(epoch) + '_' + 'conv.p')

        model.eval()
        model.evalu(test_iter)




