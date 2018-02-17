import torchtext
import torch
from torchtext.vocab import Vectors
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import visdom

from utils.models import TriGram, model_dict
from utils.preprocess import get_data, get_model
from utils.postprocess import evaluate, write_submission, vis_display


# visdom
vis_windows = None
vis = visdom.Visdom()
vis.env = 'train'

for lookback in [3]:
    model_dict['lookback'] = lookback

    train_iter, val_iter, test_iter, TEXT = get_data(model_dict)

    model = get_model(model_dict)

    trainable = False
    if len(list(model.parameters())) > 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
        trainable = True

    for epoch in range(model_dict['num_epochs']):
        for batch_num,batch in enumerate(tqdm(train_iter)):
            if trainable:
                if len(batch.text) < model_dict['lookback'] + 1:
                    continue
                model.train()
                optimizer.zero_grad()

            probs = model.train_predict(batch.text.cuda())

            if trainable:
                actuals = batch.text[-1].cuda()
                loss = F.cross_entropy(probs, actuals)
                loss.backward()
                optimizer.step()
                
                if batch_num % 100 == 0:
                    loss_l = loss.data.cpu().numpy()[0]
                    if batch_num % 250 == 0:
                        model.eval()
                        MAP = evaluate(model, val_iter)
                        print(epoch, batch_num, lookback, MAP)
                    vis_windows = vis_display(vis, vis_windows, epoch + batch_num/float(len(train_iter)), loss_l, MAP)

    model.postprocess()
    model.eval()


    MAP = evaluate(model, val_iter)
    print(model_dict)
    print("MAP FINAL", MAP)
    #write_submission(model, model_dict['output'], TEXT)



