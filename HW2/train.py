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


# NOTE: success of TriGram will depend weakly on batch size


# visdom
vis_windows = None
vis = visdom.Visdom()
vis.env = 'train'

train_iter, val_iter, test_iter, TEXT = get_data(model_dict)

model = get_model(model_dict)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

for epoch in range(model_dict['num_epochs']):
    for batch_num,batch in enumerate(tqdm(train_iter)):
        model.train()
        #optimizer.zero_grad()
        probs = model.train_predict(batch.text.cuda())

        if not probs is None:
            actuals = batch.text[-1].cuda()
            loss = F.cross_entropy(probs, actuals)
            loss.backward()
            optimizer.step()
            
            if batch_num % 200 == 0:
                loss_l = loss.data.cpu().numpy()[0]
                if batch_num % 500 == 0:
                    MAP = evaluate(model, val_iter)
                vis_windows = vis_display(vis, vis_windows, epoch + batch_num/float(len(train_iter)), loss_l, MAP)

model.postprocess()


MAP = evaluate(model, val_iter)
print("MAP", MAP, alpha)
#write_submission(model, model_dict['output'], TEXT)



