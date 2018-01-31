import torchtext
from torchtext.vocab import Vectors, GloVe
from torch.autograd import Variable

from utils.preprocess import get_data

TEXT, LABEL, train_iter, val_iter, test_iter = get_data(batch_size=10)

print (len(train_iter))
print (len(val_iter))
print (len(test_iter))

for batch_num,batch in enumerate(train_iter):
    print(batch.shape)