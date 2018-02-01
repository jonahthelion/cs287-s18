import torch
import torchtext
from torchtext.vocab import Vectors, GloVe
from torch.autograd import Variable
import torch.nn as nn

from tqdm import tqdm

from utils.preprocess import get_data
from models.psetModels import MNB



TEXT, LABEL, train_iter, val_iter, test_iter = get_data(batch_size=10)


model = MNB(V=len(TEXT.vocab), alpha=.1)

for epoch in range(1):
    for batch_num,batch in enumerate(tqdm(train_iter)):
        model.train_sample(batch.label.data - 1, batch.text.data)
model.postprocess()

for epoch in range(1):
    for batch_num,batch in enumerate(tqdm(train_iter)):
        preds = model(batch.text.data)