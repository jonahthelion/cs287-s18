import torch
import torchtext
from torchtext.vocab import Vectors, GloVe
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from sklearn import metrics

from utils.preprocess import get_data
from models.psetModels import MNB
from utils.postprocess import print_important



TEXT, LABEL, train_iter, val_iter, test_iter = get_data(batch_size=10)


model = MNB(V=len(TEXT.vocab), alpha=.1)



for epoch in range(1):
    for batch_num,batch in enumerate(tqdm(train_iter)):
        model.train_sample(batch.label.data - 1, batch.text.data)



model.postprocess()
bad_vals, bad_ixes, good_vals, good_ixes = model.find_important_words(k=10)
print_important(TEXT, bad_vals, bad_ixes, good_vals, good_ixes)


all_actual = []
all_preds = []
for epoch in range(1):
    for batch_num,batch in enumerate(train_iter):
        preds = model(batch.text.data)
        preds = F.sigmoid(preds)
        all_actual.append(batch.label.data - 1)
        all_preds.append(preds)
all_actual = torch.cat(all_actual).numpy()
all_preds = torch.cat(all_preds).numpy()

print(metrics.roc_auc_score(all_actual, all_preds))
print(metrics.classification_report(all_actual, round(all_preds)))