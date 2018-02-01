import torch
import torchtext
from torchtext.vocab import Vectors, GloVe
from torch.autograd import Variable

from tqdm import tqdm

from utils.preprocess import get_data
from models.psetModels import MNB



TEXT, LABEL, train_iter, val_iter, test_iter = get_data(batch_size=10)


model = MNB(V=len(TEXT.vocab))

for epoch in range(1):
    for batch_num,batch in enumerate(train_iter):
        print (batch_num)
        model.train_sample(batch.label.data - 1, batch.text.data)