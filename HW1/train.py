import torch
import torchtext
from torchtext.vocab import Vectors, GloVe
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from sklearn import metrics
import visdom

from utils.preprocess import get_data
from models.psetModels import MNB, LogReg
from utils.postprocess import print_important, vis_display

vis = visdom.Visdom()
vis.env = 'train'
vis_windows = {'train_bce': None}



chosen_model = {'type': 'MNB'}

TEXT, LABEL, train_iter, val_iter, test_iter = get_data(batch_size=10)



if chosen_model['type'] == 'MNB':
    model = MNB(V=len(TEXT.vocab), alpha=.1)

    for epoch in range(1):
        for batch_num,batch in enumerate(tqdm(train_iter)):
            model.train_sample(batch.label.data - 1, batch.text.data)

    model.postprocess()
    bad_vals, bad_ixes, good_vals, good_ixes = model.find_important_words(k=10)
    print_important(TEXT, bad_vals, bad_ixes, good_vals, good_ixes)

    all_actual, all_preds = model.evalu(train_iter)
    print(metrics.roc_auc_score(all_actual, all_preds))
    print(metrics.classification_report(all_actual, all_preds.round()))

    all_actual, all_preds = model.evalu(val_iter)
    print(metrics.roc_auc_score(all_actual, all_preds))
    print(metrics.classification_report(all_actual, all_preds.round()))

    print('BCE LOSS', F.binary_cross_entropy( Variable(torch.from_numpy(all_preds).float()) , Variable(torch.from_numpy(all_actual).float())))

    all_actual, all_preds = model.evalu(test_iter)
    print(metrics.roc_auc_score(all_actual, all_preds))
    print(metrics.classification_report(all_actual, all_preds.round()))

    model.submission(test_iter, 'predictions.txt')



if chosen_model['type'] == 'log_reg':
    model = LogReg(V=len(TEXT.vocab))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

    for epoch in range(10):
        for batch_num,batch in enumerate(train_iter):
            l = model.train_sample(batch.label.float() - 1, batch.text.data, optimizer)
            if batch_num % 100 != 0:
                vis_windows = vis_display(vis, vis_windows, l.data.numpy()[0], epoch + batch_num/float(len(train_iter)))
            else:
                lvals = []
                for batch in val_iter:
                    lvals.append(model.evalu_loss(batch.label.float() - 1, batch.text.data).data.numpy()[0])
                vis_windows = vis_display(vis, vis_windows, l.data.numpy()[0], epoch + batch_num/float(len(train_iter)), sum(lvals)/float(len(lvals)))