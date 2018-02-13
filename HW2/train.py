import torchtext
import torch
from torchtext.vocab import Vectors
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import visdom

from utils.models import TriGram
from utils.preprocess import get_data, get_model
from utils.postprocess import evaluate, write_submission, vis_display


# NOTE: success of TriGram will depend weakly on batch size
model_dict = {'max_size': 10001, # max is 10001
                'batch_size': 30, 
                'bptt_len': 32,
                'num_epochs': 100,

                'output': 'simple0.txt',

                # 'type': 'trigram', 
                # 'alpha': [.1, .5, .4],

                'type': 'NN',


                }

# visdom
vis_windows = None
vis = visdom.Visdom()
vis.env = 'train'

train_iter, val_iter, test_iter, TEXT = get_data(model_dict)

model = get_model(model_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=5e-4)

for epoch in range(model_dict['num_epochs']):
    for batch_num,batch in enumerate(tqdm(train_iter)):
        model.train()
        optimizer.zero_grad()
        probs = model.train_predict(batch.text.cuda())
        actuals = batch.text[-1].cuda()
        loss = F.cross_entropy(probs, actuals)
        loss.backward()
        optimizer.step()
        
        if batch_num % 20 == 0:
            loss_l = loss.data.cpu().numpy()[0]
            MAP = evaluate(model, val_iter)
            print(batch_num, loss_l)
            vis_windows = vis_display(vis, vis_windows, epoch + batch_num/float(len(train_iter)), loss_l, MAP)

model.postprocess()


MAP = evaluate(model, val_iter)
print('MAP', MAP)
