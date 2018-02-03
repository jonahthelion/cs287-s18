import torch
import torchtext
from torchtext.vocab import Vectors, GloVe
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

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

from utils.preprocess import get_data, text_to_img
from models.psetModels import MNB, LogReg, CBOW, Conv, Resnet
from utils.postprocess import print_important, vis_display, evaluate_model

# set up visdom
vis = visdom.Visdom()
vis.env = 'train'
vis_windows = None

# define model
# chosen_model = {'type': 'MNB', 'alpha':[.5], 'should_plot':False, 'counts': False, 'batch_size': 50}
# chosen_model = {'type': 'log_reg', 'batch_size': 150, 'counts':False}
# chosen_model = {'type': 'CBOW', 'batch_size': 100, 'pool': 'max'}
# chosen_model = {'type': 'Conv', 'batch_size': 50, 'embed_type': 'glove'}
chosen_model = {'type': 'resnet', 'batch_size': 2}
print(chosen_model)

# get data
TEXT, LABEL, train_iter, val_iter, test_iter = get_data(chosen_model) 

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
        print('saving', 'writeup/imgs/alpha.pdf')
        plt.savefig('writeup/imgs/alpha.pdf')
        plt.close(fig)

if chosen_model['type'] == 'log_reg':

    model = LogReg(V=len(TEXT.vocab), counts=chosen_model['counts'])
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=5e-4)

    for epoch in range(150):
        for batch_num,batch in enumerate(train_iter):
            optimizer.zero_grad()
            preds = model(batch.text.data)
            l = F.binary_cross_entropy_with_logits(preds.view(-1), (batch.label - 1).float().cuda())
            l.backward()
            optimizer.step()

            if batch_num % 40 == 0:
                bce, roc, acc = evaluate_model(model, val_iter)
                vis_windows = vis_display(vis, vis_windows, l.cpu().data.numpy()[0], epoch + batch_num/float(len(train_iter)), acc)
            if batch_num % 16 == 0 and batch_num % 40 != 0:
                vis_windows = vis_display(vis, vis_windows, l.cpu().data.numpy()[0], epoch + batch_num/float(len(train_iter)))

        print_important(model.w.weight.data.cpu().squeeze(0), TEXT, 10)
        bce, roc, acc = evaluate_model(model, test_iter)
        print('BCE:', bce, '  ROC:',roc,'  ACC:', acc)

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
    model = CBOW(V=len(TEXT.vocab), embed=TEXT.vocab.vectors, pool=chosen_model['pool'])
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00002)
    for epoch in range(350):
        for batch_num,batch in enumerate(train_iter):
            model.train()
            optimizer.zero_grad()
            preds = model(batch.text.data)
            l = F.binary_cross_entropy_with_logits(preds.view(-1), (batch.label - 1).float().cuda())
            l.backward()
            optimizer.step()
            model.eval()

            if batch_num % 160*4 == 0:
                bce, roc, acc = evaluate_model(model, val_iter)
                vis_windows = vis_display(vis, vis_windows, l.cpu().data.numpy()[0], epoch + batch_num/float(len(train_iter)), acc)
            if batch_num % 64*4 == 0 and batch_num % 160*4 != 0:
                vis_windows = vis_display(vis, vis_windows, l.cpu().data.numpy()[0], epoch + batch_num/float(len(train_iter)))
 
        bce,roc,acc = evaluate_model(model, test_iter)
        print('BCE:', bce, '  ROC:',roc,'  ACC:', acc)

if chosen_model['type'] == 'Conv':
    model = Conv(V=len(TEXT.vocab), embed=TEXT.vocab.vectors)
    model.cuda()
    optimizer = torch.optim.Adam(list(model.w3.parameters()) + list(model.w4.parameters()) + list(model.w5.parameters()) + list(model.w.parameters()), lr=0.00006) # list(model.w3.parameters()) + list(model.w4.parameters()) + list(model.w5.parameters()) + list(model.w.parameters()
    for epoch in range(350):
        for batch_num,batch in enumerate(train_iter):
            model.train()
            optimizer.zero_grad()
            preds = model(batch.text.data)
            l = F.binary_cross_entropy_with_logits(preds.view(-1), (batch.label - 1).float().cuda())
            l.backward()
            optimizer.step()
            model.eval()

            if batch_num % 160*32 == 0:
                bce, roc, acc = evaluate_model(model, val_iter)
                vis_windows = vis_display(vis, vis_windows, l.cpu().data.numpy()[0], epoch + batch_num/float(len(train_iter)), acc)
            if batch_num % 64*32 == 0 and batch_num % 160*32 != 0:
                vis_windows = vis_display(vis, vis_windows, l.cpu().data.numpy()[0], epoch + batch_num/float(len(train_iter)))
 
        bce,roc,acc = evaluate_model(model, test_iter)
        print('BCE:', bce, '  ROC:',roc,'  ACC:', acc)

if chosen_model['type'] == 'resnet':
    model = Resnet()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00006)
    all_lens = []
    for epoch in range(1):
        for batch_num,batch in enumerate(train_iter):
            model.train()
            optimizer.zero_grad()
            imgs = text_to_img(batch.text.data, TEXT)
            preds = model(imgs)
            l = F.binary_cross_entropy_with_logits(preds.view(-1), (batch.label - 1).float().cuda())
            l.backward()
            optimizer.step()
            model.eval()

            if batch_num % 160*32 == 0:
                bce, roc, acc = evaluate_model(model, val_iter)
                vis_windows = vis_display(vis, vis_windows, l.cpu().data.numpy()[0], epoch + batch_num/float(len(train_iter)), acc)
            if batch_num % 64*32 == 0 and batch_num % 160*32 != 0:
                vis_windows = vis_display(vis, vis_windows, l.cpu().data.numpy()[0], epoch + batch_num/float(len(train_iter)))



